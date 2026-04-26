const std = @import("std");

pub const VEC_SIZE: usize = 16;
pub const PREFETCH_L2: usize = 64;

const VecF32 = @Vector(VEC_SIZE, f32);

inline fn load(ptr: [*]const f32) VecF32 {
    return ptr[0..VEC_SIZE].*;
}

inline fn store(ptr: [*]f32, vec: VecF32) void {
    ptr[0..VEC_SIZE].* = vec;
}

inline fn loadF16ToF32(ptr: [*]const f16) VecF32 {
    const half: @Vector(VEC_SIZE, f16) = ptr[0..VEC_SIZE].*;
    return @floatCast(half);
}

pub inline fn accumulate(dest: [*]f32, src: [*]const f32, len: usize) void {
    const simd_end = (len / VEC_SIZE) * VEC_SIZE;
    var i: usize = 0;
    while (i < simd_end) : (i += VEC_SIZE) {
        store(dest + i, load(dest + i) + load(src + i));
    }
    while (i < len) : (i += 1) {
        dest[i] += src[i];
    }
}

pub inline fn accumulate4xF16(dest: [*]f32, src: [*]const f16, len: usize) void {
    var i: usize = 0;
    while (i + 4 * VEC_SIZE <= len) : (i += 4 * VEC_SIZE) {
        const v0 = loadF16ToF32(src + i);
        const v1 = loadF16ToF32(src + i + VEC_SIZE);
        const v2 = loadF16ToF32(src + i + 2 * VEC_SIZE);
        const v3 = loadF16ToF32(src + i + 3 * VEC_SIZE);
        store(dest + i, load(dest + i) + v0);
        store(dest + i + VEC_SIZE, load(dest + i + VEC_SIZE) + v1);
        store(dest + i + 2 * VEC_SIZE, load(dest + i + 2 * VEC_SIZE) + v2);
        store(dest + i + 3 * VEC_SIZE, load(dest + i + 3 * VEC_SIZE) + v3);
    }
    while (i + VEC_SIZE <= len) : (i += VEC_SIZE) {
        store(dest + i, load(dest + i) + loadF16ToF32(src + i));
    }
    while (i < len) : (i += 1) {
        dest[i] += @as(f32, @floatCast(src[i]));
    }
}

test "accumulate handles tails" {
    var dest = [_]f32{0} ** 19;
    var src = [_]f32{1} ** 19;
    accumulate(&dest, &src, dest.len);
    for (dest) |value| {
        try std.testing.expectEqual(@as(f32, 1), value);
    }
}
