import re

with open("src/runtime.cpp", "r") as f:
    content = f.read()

old_loop = """    for (int n = 0; n < max_new; ++n) {
        if (next < 0) break;
        if (tok_.is_eog(next)) break;
        if (pos_ >= kv_->max_ctx()) break;

        std::string piece = tok_.decode_token(next);
        output += piece;
        ++st.gen_tokens;
        if (on_token && !on_token(piece, next)) break;

        const float* logits = tf_->forward(next, pos_);
        committed_.push_back(next);
        ++pos_;
        if (rtk) rtk->on_step(kv_view());
        if (profile_sink_) profile_sink_(n, tf_->last_timings(), tf_->peak_rss());
        next = sampler.sample(logits, vocab);
    }"""

new_loop = """    for (int n = 0; n < max_new; ) {
        if (next < 0) break;
        if (tok_.is_eog(next)) break;
        if (pos_ >= kv_->max_ctx()) break;

        std::string piece = tok_.decode_token(next);
        output += piece;
        ++st.gen_tokens;
        if (on_token && !on_token(piece, next)) break;

        committed_.push_back(next);
        ++n;

        std::vector<int64_t> draft;
        if (spec_decoder_ && n < max_new && pos_ + max_draft_len_ < kv_->max_ctx()) {
            draft = spec_decoder_->draft(committed_, std::min(max_draft_len_, max_new - n));
        }

        if (draft.empty()) {
            const float* logits = tf_->forward(next, pos_);
            ++pos_;
            if (rtk) rtk->on_step(kv_view());
            if (profile_sink_) profile_sink_(n, tf_->last_timings(), tf_->peak_rss());
            next = sampler.sample(logits, vocab);
        } else {
            int K = (int)draft.size();
            std::vector<int64_t> batch;
            batch.reserve(K + 1);
            batch.push_back(next);
            batch.insert(batch.end(), draft.begin(), draft.end());

            const float* spec_logits = tf_->spec_forward(batch.data(), K + 1, pos_);
            if (rtk) rtk->on_step(kv_view());
            if (profile_sink_) profile_sink_(n, tf_->last_timings(), tf_->peak_rss());

            for (int i = 0; i <= K; ++i) {
                const float* step_logits = spec_logits + (size_t)i * vocab;
                int sampled = sampler.sample(step_logits, vocab);
                
                if (i < K) {
                    if (sampled == draft[i]) {
                        // Accept draft
                        piece = tok_.decode_token(draft[i]);
                        output += piece;
                        ++st.gen_tokens;
                        ++pos_;
                        if (on_token && !on_token(piece, draft[i])) {
                            next = -1;
                            break;
                        }
                        committed_.push_back(draft[i]);
                        ++n;
                    } else {
                        // Reject draft
                        ++pos_;
                        kv_->set_seq_len(pos_);
                        next = sampled;
                        break;
                    }
                } else {
                    // All accepted, this is the prediction for the next step
                    ++pos_;
                    next = sampled;
                }
            }
        }
    }"""

if old_loop in content:
    content = content.replace(old_loop, new_loop)
    with open("src/runtime.cpp", "w") as f:
        f.write(content)
    print("Patched successfully")
else:
    print("Could not find the old loop!")
