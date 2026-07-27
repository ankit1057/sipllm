import 'package:sipllm_flutter/sipllm_flutter.dart';

import 'model_catalog.dart';

/// Local lifecycle state of a model the app knows about — a catalog entry, an
/// imported GGUF, or an in-flight download.
enum ModelStatus { available, downloading, paused, installed, failed, importing }

/// A model tracked by the controller: its catalog metadata (if any), local
/// path, on-disk size, and current download/install state.
class ManagedModel {
  ManagedModel({
    required this.id,
    required this.name,
    required this.localPath,
    this.catalog,
    this.status = ModelStatus.available,
    this.sizeBytes = 0,
    this.progress,
    this.task,
    this.error,
    this.imported = false,
  });

  final String id;
  final String name;

  /// Absolute path the model will live at once installed.
  final String localPath;

  /// Catalog origin (null for imported / unknown files).
  final CatalogModel? catalog;

  ModelStatus status;

  /// On-disk size once installed (or approx from the catalog before).
  int sizeBytes;

  /// Live download progress, when [status] == downloading/paused.
  DownloadProgress? progress;

  /// The in-flight download task (null when not downloading).
  DownloadTask? task;

  String? error;
  final bool imported;

  bool get isInstalled => status == ModelStatus.installed;
  bool get isBusy =>
      status == ModelStatus.downloading || status == ModelStatus.importing;

  double get approxBytes =>
      (catalog?.approxBytes ?? sizeBytes).toDouble();

  double get fraction => progress?.fraction ?? (isInstalled ? 1.0 : 0.0);

  String get quant => catalog?.quant ?? 'GGUF';
  String get params => catalog?.params ?? '';
}
