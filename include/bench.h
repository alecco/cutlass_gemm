// cutlass GEMM
// Copyright (C) 2024  Ologan Ltd
// SPDX-License-Identifier: AGPL-3.0
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

//
// Benchmarking helper base class
//

#pragma once

#include <cuda_runtime_api.h>
#include <util/cuda_check.h>
#include <util/cublas_check.h>

class Bench {
public:
    Bench(const Bench&)            = delete;
    Bench& operator=(const Bench&) = delete;
    Bench();

    // Runs the benchmark
    virtual void run() = 0;
};
