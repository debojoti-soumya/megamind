// FrameSpool tests, no Unity needed: dotnet run --project headset/tests/FrameSpoolTests (exit code = failures)
using System;
using System.IO;
using System.Text;
using Megamind;

static class P
{
    static int fails;
    static void Check(bool ok, string what) { Console.WriteLine((ok ? "ok   " : "FAIL ") + what); if (!ok) fails++; }
    static byte[] B(int n) => new byte[n];

    static int Main()
    {
        var dir = Path.Combine(Path.GetTempPath(), "spool-" + Guid.NewGuid().ToString("N"));
        var s = new FrameSpool(dir, 5, 1_000_000);
        s.Enqueue(1700000002.5, Encoding.ASCII.GetBytes("b"), "{\"ts\":2.5}");
        s.Enqueue(1700000001.0, Encoding.ASCII.GetBytes("a"), "{\"ts\":1}");
        s.Enqueue(1700000002.5, Encoding.ASCII.GetBytes("c"), "{\"ts\":2.5b}");
        Check(s.Count == 3, "3 queued");
        s.TryPeek(out var st, out var j, out var m);
        Check(Encoding.ASCII.GetString(j) == "a" && m == "{\"ts\":1}", "oldest ts first");
        s.Remove(st);
        s.TryPeek(out st, out j, out _);
        Check(Encoding.ASCII.GetString(j) == "b", "equal ts keeps enqueue order");

        var r = new FrameSpool(dir, 5, 1_000_000);   // app restart
        Check(r.Count == 2 && r.Bytes == s.Bytes, "survives restart: " + r.Count);
        File.WriteAllBytes(Path.Combine(dir, FrameSpool.Stem(1600000000, 0) + ".jpg"), B(10));   // torn: jpg, no json
        File.WriteAllText(Path.Combine(dir, FrameSpool.Stem(1600000001, 0) + ".json"), "{}");   // orphan json
        r = new FrameSpool(dir, 5, 1_000_000);
        Check(r.Count == 2 && Directory.GetFiles(dir).Length == 4, "torn writes cleaned up");

        for (int i = 0; i < 10; i++) r.Enqueue(1700000100 + i, B(100), "{}");
        Check(r.Count == 5 && r.Dropped == 7, $"maxFrames keeps newest 5 (dropped {r.Dropped})");
        r.TryPeek(out _, out _, out m);
        r.TryPeek(out st, out _, out _);
        Check(st == FrameSpool.Stem(1700000105, 5), "oldest left is #5: " + st);

        var dir2 = dir + "-b";
        var b = new FrameSpool(dir2, 100, 1000);
        for (int i = 0; i < 10; i++) b.Enqueue(1700000000 + i, B(300), "{}");
        Check(b.Bytes <= 1000 && b.Count == 3, $"maxBytes cap: {b.Count} frames, {b.Bytes} B");
        while (b.TryPeek(out st, out _, out _)) b.Remove(st);
        Check(b.Count == 0 && b.Bytes == 0 && Directory.GetFiles(dir2).Length == 0, "drained to empty");
        File.Delete(Path.Combine(dir, FrameSpool.Stem(1700000105, 5) + ".jpg"));
        int before = r.Count;
        r.TryPeek(out st, out _, out _);
        Check(r.Count == before - 1 && st == FrameSpool.Stem(1700000106, 6), "unreadable entry skipped");
        var c = new FrameSpool(dir2, 100, 1_000_000);
        c.Enqueue(1700000000, B(1), "{}");
        c.Enqueue(1700003600, B(1), "{}");
        c.Enqueue(1700003601, B(1), "{}");
        Check(c.DropOlderThan(1700001800) == 1 && c.Count == 2 && c.Dropped == 1, "DropOlderThan drops yesterday's frames only");
        Check(c.DropOlderThan(1700003600) == 0, "cutoff is exclusive");
        Directory.Delete(dir, true); Directory.Delete(dir2, true);
        Console.WriteLine(fails == 0 ? "ALL PASS" : fails + " FAILED");
        return fails;
    }
}
