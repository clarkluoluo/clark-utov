package com.clarkutov.runner.test;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import com.clarkutov.runner.contract.*;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * Entry point. Two run modes:
 *
 * <ul>
 *   <li><b>demo</b> (default): one-shot showcase — runs NIST "abc" vector against
 *       libsha256.so, prints metadata / rerun / determinism / trace summary,
 *       exits. For visual inspection of the runner.</li>
 *   <li><b>serve</b>: NDJSON stdin/stdout loop driven by the clark-utov engine.
 *       One request per line, one response per line. The engine spawns this
 *       jar as a subprocess and talks to it through the {@code SubprocessRunnerAdapter}
 *       on the Python side.</li>
 * </ul>
 *
 * <p>JSON wire schema (request lines):</p>
 * <pre>
 *   {"id": &lt;int&gt;, "method": "metadata"}
 *   {"id": &lt;int&gt;, "method": "rerun",     "params": {"input_hex": &lt;hex&gt;, "observe_points": [&lt;OP&gt;...]}}
 *   {"id": &lt;int&gt;, "method": "get_trace", "params": {"input_hex": &lt;hex&gt;, "start": "0x..", "end": "0x.."}}
 *   {"id": &lt;int&gt;, "method": "shutdown"}
 * </pre>
 */
public class Main {

    private static final String SO_PATH_DEFAULT = "../libs/arm64-v8a/libsha256.so";
    private static final String EXPECTED_ABC =
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

    public static void main(String[] args) {
        String mode = (args.length > 0) ? args[0] : "demo";
        String soPath = (args.length > 1) ? args[1] : SO_PATH_DEFAULT;
        File soFile = new File(soPath);
        if (!soFile.exists()) {
            System.err.println("error: .so not found: " + soFile.getAbsolutePath());
            System.exit(2);
        }
        try (Sha256TestRunner runner = new Sha256TestRunner(soFile)) {
            if ("serve".equals(mode)) {
                serve(runner);
            } else {
                demo(runner);
            }
        } catch (Exception e) {
            System.err.println("runner crashed:");
            e.printStackTrace(System.err);
            System.exit(1);
        }
    }

    // ---------------- demo mode ----------------

    private static void demo(Sha256TestRunner runner) {
        TargetMeta meta = runner.metadata();
        section("metadata()");
        System.out.println("  " + meta);

        section("rerun(\"abc\") + observe regs at algoEntryPc");
        ObservePoint obs = ObservePoint.regsAfter(meta.algoEntryPc,
                List.of("x0", "x1", "x2", "sp", "lr"));
        RerunResult r = runner.rerun("abc".getBytes(), List.of(obs));
        String digest = bytesToHex(r.output);
        boolean ok = EXPECTED_ABC.equals(digest);
        System.out.println("  digest:    " + digest);
        System.out.println("  expected:  " + EXPECTED_ABC);
        System.out.println("  match:     " + (ok ? "PASS" : "FAIL"));
        if (!r.observations.isEmpty()) {
            ObservedState s = r.observations.get(0);
            System.out.println("  observed @ 0x" + Long.toHexString(s.pc)
                    + " (when=" + s.when + ")");
            for (Map.Entry<String, Long> e : s.regs.entrySet()) {
                System.out.printf("    %-4s = 0x%x%n", e.getKey(), e.getValue());
            }
        }

        section("determinism: rerun(\"abc\") × 3");
        byte[] first = null;
        boolean allEqual = true;
        for (int i = 0; i < 3; i++) {
            RerunResult rr = runner.rerun("abc".getBytes(), Collections.emptyList());
            if (first == null) first = rr.output;
            else if (!Arrays.equals(first, rr.output)) allEqual = false;
        }
        System.out.println("  3 reruns produce identical output: " + (allEqual ? "PASS" : "FAIL"));

        section("getTrace(\"abc\")");
        try (TraceStream ts = runner.getTrace("abc".getBytes(),
                meta.algoEntryPc, meta.algoExitPc)) {
            System.out.println("  format:    " + ts.format());
            System.out.println("  path:      " + ts.path());
            if (ts instanceof FileTraceStream f) {
                try { System.out.println("  lines:     " + f.lineCount()); }
                catch (IOException ignored) { /* skip */ }
            }
        }
        if (!ok || !allEqual) System.exit(1);
    }

    private static void section(String title) {
        System.out.println();
        System.out.println("=== " + title + " ===");
    }

    // ---------------- serve mode (NDJSON loop) ----------------

    private static void serve(Sha256TestRunner runner) throws IOException {
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
        // Print a ready banner on stderr so the parent process can sync; protocol uses stdout only.
        System.err.println("runner ready");
        System.err.flush();
        String line;
        while ((line = in.readLine()) != null) {
            if (line.isEmpty()) continue;
            JSONObject req = JSON.parseObject(line);
            Object id = req.get("id");
            String method = req.getString("method");
            try {
                Object result = dispatch(runner, method, req.getJSONObject("params"));
                JSONObject resp = new JSONObject(true);   // ordered
                resp.put("id", id);
                resp.put("result", result);
                System.out.println(resp.toJSONString());
                System.out.flush();
                if ("shutdown".equals(method)) return;
            } catch (Exception e) {
                JSONObject err = new JSONObject(true);
                err.put("id", id);
                JSONObject errBody = new JSONObject(true);
                errBody.put("code", -32000);
                errBody.put("message", e.getClass().getSimpleName() + ": " + e.getMessage());
                err.put("error", errBody);
                System.out.println(err.toJSONString());
                System.out.flush();
            }
        }
    }

    private static Object dispatch(Sha256TestRunner runner, String method, JSONObject params) {
        switch (method) {
            case "metadata":
                return metaToJson(runner.metadata());
            case "rerun":
                return rerunToJson(runner, params);
            case "get_trace":
                return getTraceToJson(runner, params);
            case "shutdown":
                return "ok";
            default:
                throw new IllegalArgumentException("unknown method: " + method);
        }
    }

    private static JSONObject metaToJson(TargetMeta m) {
        JSONObject o = new JSONObject(true);
        o.put("target_name",  m.targetName);
        o.put("arch",         m.arch);
        o.put("algo_entry_pc", hexOf(m.algoEntryPc));
        o.put("algo_exit_pc",  hexOf(m.algoExitPc));
        o.put("input_length",  m.inputLength);
        o.put("output_length", m.outputLength);
        o.put("algo_symbol",   m.algoSymbol);
        o.put("emulator_name",    m.emulatorName);
        o.put("emulator_version", m.emulatorVersion);
        return o;
    }

    private static JSONObject rerunToJson(Sha256TestRunner runner, JSONObject params) {
        byte[] input = hexToBytes(params.getString("input_hex"));
        List<ObservePoint> ops = parseObservePoints(params.getJSONArray("observe_points"));
        RerunResult r = runner.rerun(input, ops);
        JSONObject o = new JSONObject(true);
        o.put("output_hex", bytesToHex(r.output));
        JSONArray obs = new JSONArray();
        for (ObservedState s : r.observations) {
            JSONObject so = new JSONObject(true);
            so.put("pc",   hexOf(s.pc));
            so.put("when", s.when.name());
            JSONObject regs = new JSONObject(true);
            for (var e : s.regs.entrySet()) regs.put(e.getKey(), hexOf(e.getValue()));
            so.put("regs", regs);
            JSONObject mems = new JSONObject(true);
            for (var e : s.mem.entrySet()) mems.put(hexOf(e.getKey()), bytesToHex(e.getValue()));
            so.put("mem", mems);
            obs.add(so);
        }
        o.put("observations", obs);
        return o;
    }

    private static JSONObject getTraceToJson(Sha256TestRunner runner, JSONObject params) {
        byte[] input = hexToBytes(params.getString("input_hex"));
        long start = parseHexOrDec(params.getString("start"));
        long end   = parseHexOrDec(params.getString("end"));
        try (TraceStream ts = runner.getTrace(input, start, end)) {
            JSONObject o = new JSONObject(true);
            o.put("trace_path", ts.path().toString());
            o.put("format",     ts.format());
            return o;
        }
    }

    private static List<ObservePoint> parseObservePoints(JSONArray arr) {
        if (arr == null) return Collections.emptyList();
        List<ObservePoint> out = new ArrayList<>(arr.size());
        for (int i = 0; i < arr.size(); i++) {
            JSONObject o = arr.getJSONObject(i);
            long pc = parseHexOrDec(o.getString("pc"));
            ObservePoint.When when = ObservePoint.When.valueOf(o.getString("when"));
            JSONArray regsArr = o.getJSONArray("regs");
            List<String> regs = new ArrayList<>();
            if (regsArr != null) for (int j = 0; j < regsArr.size(); j++) regs.add(regsArr.getString(j));
            EnumSet<ObservePoint.Capture> cap = EnumSet.noneOf(ObservePoint.Capture.class);
            cap.add(ObservePoint.Capture.REGS);
            // mem ranges optional — skipped for now (the Python side doesn't yet request them in C1-C4)
            out.add(new ObservePoint(pc, when, cap, regs, Collections.emptyList()));
        }
        return out;
    }

    // ---------------- helpers ----------------

    private static String bytesToHex(byte[] bs) {
        StringBuilder sb = new StringBuilder(bs.length * 2);
        for (byte b : bs) sb.append(String.format("%02x", b & 0xff));
        return sb.toString();
    }

    private static byte[] hexToBytes(String hex) {
        if (hex == null || hex.isEmpty()) return new byte[0];
        if (hex.startsWith("0x") || hex.startsWith("0X")) hex = hex.substring(2);
        int n = hex.length() / 2;
        byte[] out = new byte[n];
        for (int i = 0; i < n; i++) {
            out[i] = (byte) Integer.parseInt(hex.substring(i * 2, i * 2 + 2), 16);
        }
        return out;
    }

    private static long parseHexOrDec(String s) {
        if (s.startsWith("0x") || s.startsWith("0X")) return Long.parseLong(s.substring(2), 16);
        return Long.parseLong(s);
    }

    private static String hexOf(long v) {
        return "0x" + Long.toHexString(v);
    }
}
