const std = @import("std");

const MAX_THREADS: usize = 64;
const DEFAULT_MIN_ITEMS_PER_THREAD: usize = 64;

var configured_num_threads = std.atomic.Value(usize).init(0);

export fn tropnn_set_num_threads(num_threads: usize) void {
    configured_num_threads.store(@min(num_threads, MAX_THREADS), .monotonic);
}

pub export fn tropnn_get_num_threads() usize {
    const requested = configured_num_threads.load(.monotonic);
    if (requested != 0) return requested;
    return @min(std.Thread.getCpuCount() catch 1, MAX_THREADS);
}

inline fn workerCount(item_count: usize, min_items_per_thread: usize) usize {
    const min_items = @max(@as(usize, 1), min_items_per_thread);
    if (item_count <= min_items) return 1;
    const requested = tropnn_get_num_threads();
    if (requested <= 1) return 1;
    const useful_threads = (item_count + min_items - 1) / min_items;
    return @max(@as(usize, 1), @min(@min(requested, useful_threads), item_count));
}

pub fn parallelFor(
    comptime Context: type,
    context: *Context,
    item_count: usize,
    min_items_per_thread: usize,
    comptime workerFn: fn (*Context, usize, usize) void,
) void {
    const thread_count = workerCount(item_count, min_items_per_thread);
    if (thread_count <= 1) {
        workerFn(context, 0, item_count);
        return;
    }

    var threads: [MAX_THREADS]std.Thread = undefined;
    var spawned: usize = 0;
    var start: usize = 0;
    var worker: usize = 0;
    while (worker < thread_count) : (worker += 1) {
        const end = (item_count * (worker + 1)) / thread_count;
        if (worker + 1 == thread_count) {
            workerFn(context, start, end);
        } else {
            const thread = std.Thread.spawn(.{}, workerFn, .{ context, start, end }) catch {
                workerFn(context, start, end);
                start = end;
                continue;
            };
            threads[spawned] = thread;
            spawned += 1;
        }
        start = end;
    }

    for (threads[0..spawned]) |thread| {
        thread.join();
    }
}

test "worker count respects configuration" {
    tropnn_set_num_threads(1);
    try std.testing.expectEqual(@as(usize, 1), workerCount(1024, DEFAULT_MIN_ITEMS_PER_THREAD));
    tropnn_set_num_threads(4);
    try std.testing.expectEqual(@as(usize, 4), workerCount(1024, DEFAULT_MIN_ITEMS_PER_THREAD));
    try std.testing.expectEqual(@as(usize, 1), workerCount(8, DEFAULT_MIN_ITEMS_PER_THREAD));
    try std.testing.expectEqual(@as(usize, 2), workerCount(128, DEFAULT_MIN_ITEMS_PER_THREAD));
    tropnn_set_num_threads(0);
}
