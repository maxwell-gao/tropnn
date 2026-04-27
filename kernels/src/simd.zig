const std = @import("std");

pub const VEC_SIZE: usize = 16;
pub const PREFETCH_L2: usize = 64;

pub const VecF32 = @Vector(VEC_SIZE, f32);

pub inline fn load(ptr: [*]const f32) VecF32 {
    return ptr[0..VEC_SIZE].*;
}

inline fn store(ptr: [*]f32, vec: VecF32) void {
    ptr[0..VEC_SIZE].* = vec;
}

pub inline fn loadF16ToF32(ptr: [*]const f16) VecF32 {
    const half: @Vector(VEC_SIZE, f16) = ptr[0..VEC_SIZE].*;
    return @floatCast(half);
}

pub inline fn zero(dest: [*]f32, len: usize) void {
    const zeros: VecF32 = @splat(0.0);
    const simd_end = (len / VEC_SIZE) * VEC_SIZE;
    var i: usize = 0;
    while (i < simd_end) : (i += VEC_SIZE) {
        store(dest + i, zeros);
    }
    while (i < len) : (i += 1) {
        dest[i] = 0.0;
    }
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

pub inline fn accumulate4x(dest: [*]f32, src: [*]const f32, len: usize) void {
    var i: usize = 0;
    while (i + 4 * VEC_SIZE <= len) : (i += 4 * VEC_SIZE) {
        store(dest + i, load(dest + i) + load(src + i));
        store(dest + i + VEC_SIZE, load(dest + i + VEC_SIZE) + load(src + i + VEC_SIZE));
        store(dest + i + 2 * VEC_SIZE, load(dest + i + 2 * VEC_SIZE) + load(src + i + 2 * VEC_SIZE));
        store(dest + i + 3 * VEC_SIZE, load(dest + i + 3 * VEC_SIZE) + load(src + i + 3 * VEC_SIZE));
    }
    while (i + VEC_SIZE <= len) : (i += VEC_SIZE) {
        store(dest + i, load(dest + i) + load(src + i));
    }
    while (i < len) : (i += 1) {
        dest[i] += src[i];
    }
}

pub inline fn accumulate4Sources(
    dest: [*]f32,
    src0: [*]const f32,
    src1: [*]const f32,
    src2: [*]const f32,
    src3: [*]const f32,
    len: usize,
) void {
    var i: usize = 0;
    while (i + VEC_SIZE <= len) : (i += VEC_SIZE) {
        const sum = load(src0 + i) + load(src1 + i) + load(src2 + i) + load(src3 + i);
        store(dest + i, load(dest + i) + sum);
    }
    while (i < len) : (i += 1) {
        dest[i] += src0[i] + src1[i] + src2[i] + src3[i];
    }
}

pub inline fn copy(dest: [*]f32, src: [*]const f32, len: usize) void {
    const simd_end = (len / VEC_SIZE) * VEC_SIZE;
    var i: usize = 0;
    while (i < simd_end) : (i += VEC_SIZE) {
        store(dest + i, load(src + i));
    }
    while (i < len) : (i += 1) {
        dest[i] = src[i];
    }
}

pub inline fn addScaled(dest: [*]f32, src: [*]const f32, scale: f32, len: usize) void {
    const scale_vec: VecF32 = @splat(scale);
    const simd_end = (len / VEC_SIZE) * VEC_SIZE;
    var i: usize = 0;
    while (i < simd_end) : (i += VEC_SIZE) {
        store(dest + i, load(dest + i) + load(src + i) * scale_vec);
    }
    while (i < len) : (i += 1) {
        dest[i] += src[i] * scale;
    }
}

pub inline fn addScaledF16(dest: [*]f32, src: [*]const f16, scale: f32, len: usize) void {
    const scale_vec: VecF32 = @splat(scale);
    var i: usize = 0;
    while (i + VEC_SIZE <= len) : (i += VEC_SIZE) {
        store(dest + i, load(dest + i) + loadF16ToF32(src + i) * scale_vec);
    }
    while (i < len) : (i += 1) {
        dest[i] += @as(f32, @floatCast(src[i])) * scale;
    }
}

pub inline fn dot(a: [*]const f32, b: [*]const f32, len: usize) f32 {
    var acc: VecF32 = @splat(0.0);
    const simd_end = (len / VEC_SIZE) * VEC_SIZE;
    var i: usize = 0;
    while (i < simd_end) : (i += VEC_SIZE) {
        acc += load(a + i) * load(b + i);
    }
    var total: f32 = @reduce(.Add, acc);
    while (i < len) : (i += 1) {
        total += a[i] * b[i];
    }
    return total;
}

pub inline fn dotF16(a: [*]const f32, b: [*]const f16, len: usize) f32 {
    var acc: VecF32 = @splat(0.0);
    const simd_end = (len / VEC_SIZE) * VEC_SIZE;
    var i: usize = 0;
    while (i < simd_end) : (i += VEC_SIZE) {
        acc += load(a + i) * loadF16ToF32(b + i);
    }
    var total: f32 = @reduce(.Add, acc);
    while (i < len) : (i += 1) {
        total += a[i] * @as(f32, @floatCast(b[i]));
    }
    return total;
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

pub inline fn accumulate4SourcesF16(
    dest: [*]f32,
    src0: [*]const f16,
    src1: [*]const f16,
    src2: [*]const f16,
    src3: [*]const f16,
    len: usize,
) void {
    var i: usize = 0;
    while (i + VEC_SIZE <= len) : (i += VEC_SIZE) {
        const sum = loadF16ToF32(src0 + i) + loadF16ToF32(src1 + i) + loadF16ToF32(src2 + i) + loadF16ToF32(src3 + i);
        store(dest + i, load(dest + i) + sum);
    }
    while (i < len) : (i += 1) {
        dest[i] += @as(f32, @floatCast(src0[i])) + @as(f32, @floatCast(src1[i])) + @as(f32, @floatCast(src2[i])) + @as(f32, @floatCast(src3[i]));
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

test "dot and scaled add handle tails" {
    var dest = [_]f32{1} ** 19;
    var a = [_]f32{2} ** 19;
    var b = [_]f32{3} ** 19;
    try std.testing.expectEqual(@as(f32, 114), dot(&a, &b, a.len));
    addScaled(&dest, &a, 0.5, dest.len);
    for (dest) |value| {
        try std.testing.expectEqual(@as(f32, 2), value);
    }
}

test "grouped accumulators handle tails" {
    var dest = [_]f32{1} ** 19;
    var a = [_]f32{2} ** 19;
    var b = [_]f32{3} ** 19;
    var c = [_]f32{4} ** 19;
    var d = [_]f32{5} ** 19;
    accumulate4Sources(&dest, &a, &b, &c, &d, dest.len);
    for (dest) |value| {
        try std.testing.expectEqual(@as(f32, 15), value);
    }

    var half_dest = [_]f32{1} ** 19;
    var ha = [_]f16{@as(f16, @floatCast(2.0))} ** 19;
    var hb = [_]f16{@as(f16, @floatCast(3.0))} ** 19;
    var hc = [_]f16{@as(f16, @floatCast(4.0))} ** 19;
    var hd = [_]f16{@as(f16, @floatCast(5.0))} ** 19;
    accumulate4SourcesF16(&half_dest, &ha, &hb, &hc, &hd, half_dest.len);
    for (half_dest) |value| {
        try std.testing.expectEqual(@as(f32, 15), value);
    }
}
