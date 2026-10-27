/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#import "ETCoreAIBridge.h"
#import "coreai_bookmarks.h"

namespace executorch::backends::coreai {

struct RuntimeAssets {
  Manifest manifest;
  NSString* platform;
  NSString* architecture;
};

// Reuse a resolved architecture when selecting multiple delegates in one PTE.
runtime::Result<RuntimeAssets> select_runtime_assets(
    const Manifest& manifest,
    NSString* architecture = nil);

// The key lock outlives all SDK callbacks and raw bookmark publication.
// Cleanup after a restore hit or durable publication is best-effort.
runtime::Result<id<ETCoreAIPreparedModel>> acquire_bookmark_model(
    const Manifest& selected,
    const runtime::NamedDataMap* named_data,
    NSString* root,
    NSString* platform,
    NSString* architecture,
    id<ETCoreAIModelLoading> loader);

// Callers must keep affected loads and sessions quiescent across processes.
runtime::Error clear_bookmark_assets(
    NSString* root,
    NSString* key,
    id<ETCoreAIModelLoading> loader);

} // namespace executorch::backends::coreai
