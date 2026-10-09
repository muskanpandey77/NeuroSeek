import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.UnsupportedEncodingException;
import java.net.InetSocketAddress;
import java.net.HttpURLConnection;
import java.net.URI;
import java.net.URL;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.Executors;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * VectorDB — Java port of main.cpp
 * HNSW + KD-Tree + Brute Force, REST API, Ollama-powered RAG.
 * Zero external dependencies (JDK 8+).
 */
public class Main {

    static final int DIMS = 16; // demo vectors
    // Doc embeddings dimension is determined at runtime from Ollama's model output

    // =====================================================================
    // DATA TYPES
    // =====================================================================
    static final class VectorItem {
        final int id;
        final String metadata;
        final String category;
        final float[] emb;

        VectorItem(int id, String metadata, String category, float[] emb) {
            this.id = id;
            this.metadata = metadata;
            this.category = category;
            this.emb = emb;
        }
    }

    /** std::pair<float,int> equivalent (ordered by dist, then id). */
    static final class Pair implements Comparable<Pair> {
        final float dist;
        final int id;

        Pair(float dist, int id) {
            this.dist = dist;
            this.id = id;
        }

        @Override
        public int compareTo(Pair o) {
            int c = Float.compare(dist, o.dist);
            return c != 0 ? c : Integer.compare(id, o.id);
        }
    }

    interface DistFn {
        float apply(float[] a, float[] b);
    }

    // =====================================================================
    // DISTANCE METRICS
    // =====================================================================
    static float euclidean(float[] a, float[] b) {
        float s = 0;
        for (int i = 0; i < a.length; i++) {
            float d = a[i] - b[i];
            s += d * d;
        }
        return (float) Math.sqrt(s);
    }

    static float cosine(float[] a, float[] b) {
        float dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        if (na < 1e-9f || nb < 1e-9f) return 1.0f;
        return 1.0f - dot / ((float) Math.sqrt(na) * (float) Math.sqrt(nb));
    }

    static float manhattan(float[] a, float[] b) {
        float s = 0;
        for (int i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
        return s;
    }

    static DistFn getDistFn(String m) {
        if (m.equals("cosine")) return Main::cosine;
        if (m.equals("manhattan")) return Main::manhattan;
        return Main::euclidean;
    }

    // =====================================================================
    // BRUTE FORCE
    // =====================================================================
    static final class BruteForce {
        final List<VectorItem> items = new ArrayList<>();

        void insert(VectorItem v) {
            items.add(v);
        }

        List<Pair> knn(float[] q, int k, DistFn dist) {
            List<Pair> r = new ArrayList<>(items.size());
            for (VectorItem v : items) r.add(new Pair(dist.apply(q, v.emb), v.id));
            Collections.sort(r);
            if (r.size() > k) r = new ArrayList<>(r.subList(0, Math.max(k, 0)));
            return r;
        }

        void remove(int id) {
            items.removeIf(v -> v.id == id);
        }
    }

    // =====================================================================
    // KD-TREE
    // =====================================================================
    static final class KDNode {
        final VectorItem item;
        KDNode left, right;

        KDNode(VectorItem v) {
            this.item = v;
        }
    }

    static final class KDTree {
        private KDNode root = null;
        private final int dims;

        KDTree(int d) {
            this.dims = d;
        }

        private KDNode ins(KDNode n, VectorItem v, int d) {
            if (n == null) return new KDNode(v);
            int ax = d % dims;
            if (v.emb[ax] < n.item.emb[ax]) n.left = ins(n.left, v, d + 1);
            else n.right = ins(n.right, v, d + 1);
            return n;
        }

        private void knn(KDNode n, float[] q, int k, int d, DistFn dist, PriorityQueue<Pair> heap) {
            if (n == null) return;
            float dn = dist.apply(q, n.item.emb);
            if (heap.size() < k || dn < heap.peek().dist) {
                heap.add(new Pair(dn, n.item.id));
                if (heap.size() > k) heap.poll();
            }
            int ax = d % dims;
            float diff = q[ax] - n.item.emb[ax];
            KDNode closer = diff < 0 ? n.left : n.right;
            KDNode farther = diff < 0 ? n.right : n.left;
            knn(closer, q, k, d + 1, dist, heap);
            if (heap.size() < k || Math.abs(diff) < heap.peek().dist)
                knn(farther, q, k, d + 1, dist, heap);
        }

        void insert(VectorItem v) {
            root = ins(root, v, 0);
        }

        List<Pair> knn(float[] q, int k, DistFn dist) {
            if (k <= 0) return new ArrayList<>();
            PriorityQueue<Pair> heap = new PriorityQueue<>(Collections.reverseOrder());
            knn(root, q, k, 0, dist, heap);
            List<Pair> r = new ArrayList<>();
            while (!heap.isEmpty()) r.add(heap.poll());
            Collections.sort(r);
            return r;
        }

        void rebuild(Collection<VectorItem> items) {
            root = null;
            for (VectorItem v : items) insert(v);
        }
    }

    // =====================================================================
    // MT19937 — same generator as C++ std::mt19937 (seed 42), so the HNSW
    // layer assignment matches the original implementation.
    // =====================================================================
    static final class MT19937 {
        private final int[] mt = new int[624];
        private int idx;

        MT19937(int seed) {
            mt[0] = seed;
            for (int i = 1; i < 624; i++)
                mt[i] = 1812433253 * (mt[i - 1] ^ (mt[i - 1] >>> 30)) + i;
            idx = 624;
        }

        private void twist() {
            for (int i = 0; i < 624; i++) {
                int y = (mt[i] & 0x80000000) | (mt[(i + 1) % 624] & 0x7fffffff);
                mt[i] = mt[(i + 397) % 624] ^ (y >>> 1) ^ ((y & 1) != 0 ? 0x9908b0df : 0);
            }
            idx = 0;
        }

        int nextInt() {
            if (idx >= 624) twist();
            int y = mt[idx++];
            y ^= (y >>> 11);
            y ^= (y << 7) & 0x9d2c5680;
            y ^= (y << 15) & 0xefc60000;
            y ^= (y >>> 18);
            return y;
        }

        /** uniform_real_distribution<float>(0,1) as implemented by libstdc++ */
        float nextFloat() {
            long x = nextInt() & 0xFFFFFFFFL;
            float u = (float) x / 4294967296.0f;
            if (u >= 1.0f) u = Math.nextDown(1.0f);
            return u;
        }
    }

    // =====================================================================
    // StdIntMap — reproduces the iteration order of libstdc++'s
    // std::unordered_map<int, V> (the C++ original iterates these maps, so
    // the order in which items/documents/graph nodes are listed depends on
    // it). Same bucket growth, same rehash and same insert/erase behaviour.
    // =====================================================================
    static final class StdIntMap<V> {
        private static final class N<V> {
            final int key;
            V val;
            N<V> next;

            N(int key, V val) {
                this.key = key;
                this.val = val;
            }
        }

        // bucket counts libstdc++ walks through (1 -> 13 -> 29 -> 59 -> ...)
        private static final int[] PRIMES = {13, 29, 59, 127, 257, 541, 1109, 2357, 5087, 10273,
                20753, 42043, 85229, 172933, 351061, 712697, 1447153, 2938679, 5967347};
        private static final int[] FAST_BKT = {2, 2, 2, 3, 5, 5, 7, 7, 11, 11, 11, 11, 13, 13};

        private final N<V> before = new N<>(0, null); // _M_before_begin
        private N<V>[] buckets = newBuckets(1);        // each entry = node *before* the bucket's first node
        private int nBkt = 1;
        private int count = 0;
        private int nextResize = 0;
        private final HashMap<Integer, N<V>> index = new HashMap<>();

        @SuppressWarnings("unchecked")
        private N<V>[] newBuckets(int n) {
            return (N<V>[]) new N[n];
        }

        private int bkt(int key, int n) {
            return Integer.remainderUnsigned(key, n);
        }

        private int nextBkt(int n) {
            if (n < FAST_BKT.length) {
                if (n == 0) return 1;
                nextResize = FAST_BKT[n];
                return FAST_BKT[n];
            }
            for (int p : PRIMES) {
                if (p >= n) {
                    nextResize = p;
                    return p;
                }
            }
            nextResize = n;
            return n;
        }

        private void rehash(int n) {
            N<V>[] nb = newBuckets(n);
            N<V> p = before.next;
            before.next = null;
            int bbeginBkt = 0;
            while (p != null) {
                N<V> nx = p.next;
                int b = bkt(p.key, n);
                if (nb[b] == null) {
                    p.next = before.next;
                    before.next = p;
                    nb[b] = before;
                    if (p.next != null) nb[bbeginBkt] = p;
                    bbeginBkt = b;
                } else {
                    p.next = nb[b].next;
                    nb[b].next = p;
                }
                p = nx;
            }
            buckets = nb;
            nBkt = n;
        }

        V put(int key, V val) {
            N<V> ex = index.get(key);
            if (ex != null) {
                V old = ex.val;
                ex.val = val;
                return old;
            }
            if (count + 1 > nextResize) {
                double minBkts = Math.max(count + 1, nextResize == 0 ? 11 : 0);
                if (minBkts >= nBkt) {
                    rehash(nextBkt((int) Math.max(Math.floor(minBkts) + 1, nBkt * 2.0)));
                } else {
                    nextResize = nBkt;
                }
            }
            N<V> node = new N<>(key, val);
            int b = bkt(key, nBkt);
            if (buckets[b] != null) {
                node.next = buckets[b].next;
                buckets[b].next = node;
            } else {
                node.next = before.next;
                before.next = node;
                if (node.next != null) buckets[bkt(node.next.key, nBkt)] = node;
                buckets[b] = before;
            }
            index.put(key, node);
            count++;
            return null;
        }

        V get(int key) {
            N<V> n = index.get(key);
            return n == null ? null : n.val;
        }

        boolean containsKey(int key) {
            return index.containsKey(key);
        }

        V remove(int key) {
            N<V> n = index.remove(key);
            if (n == null) return null;
            int b = bkt(key, nBkt);
            N<V> prev = buckets[b];
            while (prev.next != n) prev = prev.next;
            if (prev == buckets[b]) {
                N<V> nx = n.next;
                int nxBkt = nx != null ? bkt(nx.key, nBkt) : 0;
                if (nx == null || nxBkt != b) {
                    if (nx != null) buckets[nxBkt] = buckets[b];
                    if (buckets[b] == before) before.next = nx;
                    buckets[b] = null;
                }
            } else if (n.next != null) {
                int nxBkt = bkt(n.next.key, nBkt);
                if (nxBkt != b) buckets[nxBkt] = prev;
            }
            prev.next = n.next;
            count--;
            return n.val;
        }

        int size() {
            return count;
        }

        boolean isEmpty() {
            return count == 0;
        }

        List<Integer> keySet() {
            List<Integer> r = new ArrayList<>(count);
            for (N<V> n = before.next; n != null; n = n.next) r.add(n.key);
            return r;
        }

        List<V> values() {
            List<V> r = new ArrayList<>(count);
            for (N<V> n = before.next; n != null; n = n.next) r.add(n.val);
            return r;
        }
    }

    // =====================================================================
    // HNSW — Hierarchical Navigable Small World
    // =====================================================================
    static final class HNSW {
        static final class Node {
            VectorItem item;
            int maxLyr;
            List<List<Integer>> nbrs;

            Node(VectorItem item, int maxLyr, List<List<Integer>> nbrs) {
                this.item = item;
                this.maxLyr = maxLyr;
                this.nbrs = nbrs;
            }
        }

        private final StdIntMap<Node> G = new StdIntMap<>();
        private final int M, M0, ef_build;
        private final float mL;
        private int topLayer = -1;
        private int entryPt = -1;
        private final MT19937 rng;

        HNSW(int m, int efBuild) {
            this.M = m;
            this.M0 = 2 * m;
            this.ef_build = efBuild;
            this.mL = 1.0f / (float) Math.log((float) m);
            this.rng = new MT19937(42);
        }

        HNSW() {
            this(16, 200);
        }

        private int randLevel() {
            float u = rng.nextFloat();
            return (int) Math.floor((float) (-Math.log(u)) * mL);
        }

        private List<Pair> searchLayer(float[] q, int ep, int ef, int lyr, DistFn dist) {
            Set<Integer> vis = new HashSet<>();
            PriorityQueue<Pair> cands = new PriorityQueue<>();                          // min-heap
            PriorityQueue<Pair> found = new PriorityQueue<>(Collections.reverseOrder()); // max-heap

            float d0 = dist.apply(q, G.get(ep).item.emb);
            vis.add(ep);
            cands.add(new Pair(d0, ep));
            found.add(new Pair(d0, ep));

            while (!cands.isEmpty()) {
                Pair c = cands.poll();
                if (found.size() >= ef && c.dist > found.peek().dist) break;
                Node cn = G.get(c.id);
                if (lyr >= cn.nbrs.size()) continue;
                for (int nid : cn.nbrs.get(lyr)) {
                    if (vis.contains(nid) || !G.containsKey(nid)) continue;
                    vis.add(nid);
                    float nd = dist.apply(q, G.get(nid).item.emb);
                    if (found.size() < ef || nd < found.peek().dist) {
                        cands.add(new Pair(nd, nid));
                        found.add(new Pair(nd, nid));
                        if (found.size() > ef) found.poll();
                    }
                }
            }
            List<Pair> res = new ArrayList<>();
            while (!found.isEmpty()) res.add(found.poll());
            Collections.sort(res);
            return res;
        }

        private List<Integer> selectNbrs(List<Pair> cands, int maxM) {
            List<Integer> r = new ArrayList<>();
            for (int i = 0; i < Math.min(cands.size(), maxM); i++) r.add(cands.get(i).id);
            return r;
        }

        void insert(VectorItem item, DistFn dist) {
            int id = item.id;
            int lvl = randLevel();
            List<List<Integer>> layers = new ArrayList<>();
            for (int i = 0; i <= lvl; i++) layers.add(new ArrayList<>());
            G.put(id, new Node(item, lvl, layers));

            if (entryPt == -1) {
                entryPt = id;
                topLayer = lvl;
                return;
            }

            int ep = entryPt;
            for (int lc = topLayer; lc > lvl; lc--) {
                if (lc < G.get(ep).nbrs.size()) {
                    List<Pair> W = searchLayer(item.emb, ep, 1, lc, dist);
                    if (!W.isEmpty()) ep = W.get(0).id;
                }
            }

            for (int lc = Math.min(topLayer, lvl); lc >= 0; lc--) {
                List<Pair> W = searchLayer(item.emb, ep, ef_build, lc, dist);
                int maxM = (lc == 0) ? M0 : M;
                List<Integer> sel = selectNbrs(W, maxM);
                G.get(id).nbrs.set(lc, new ArrayList<>(sel));

                for (int nid : sel) {
                    Node nn = G.get(nid);
                    if (nn == null) continue;
                    while (nn.nbrs.size() <= lc) nn.nbrs.add(new ArrayList<>());
                    List<Integer> conn = nn.nbrs.get(lc);
                    conn.add(id);
                    if (conn.size() > maxM) {
                        List<Pair> ds = new ArrayList<>();
                        for (int c : conn) {
                            Node cn = G.get(c);
                            if (cn != null) ds.add(new Pair(dist.apply(nn.item.emb, cn.item.emb), c));
                        }
                        Collections.sort(ds);
                        conn.clear();
                        for (int i = 0; i < maxM && i < ds.size(); i++) conn.add(ds.get(i).id);
                    }
                }
                if (!W.isEmpty()) ep = W.get(0).id;
            }

            if (lvl > topLayer) {
                topLayer = lvl;
                entryPt = id;
            }
        }

        List<Pair> knn(float[] q, int k, int ef, DistFn dist) {
            if (entryPt == -1) return new ArrayList<>();
            int ep = entryPt;
            for (int lc = topLayer; lc > 0; lc--) {
                if (lc < G.get(ep).nbrs.size()) {
                    List<Pair> W = searchLayer(q, ep, 1, lc, dist);
                    if (!W.isEmpty()) ep = W.get(0).id;
                }
            }
            List<Pair> W = searchLayer(q, ep, Math.max(ef, k), 0, dist);
            if (W.size() > k) W = new ArrayList<>(W.subList(0, Math.max(k, 0)));
            return W;
        }

        void remove(int id) {
            if (!G.containsKey(id)) return;
            for (Node nd : G.values())
                for (List<Integer> layer : nd.nbrs)
                    layer.removeIf(x -> x == id);
            if (entryPt == id) {
                entryPt = -1;
                for (Integer nid : G.keySet()) {
                    if (nid != id) {
                        entryPt = nid;
                        break;
                    }
                }
            }
            G.remove(id);
        }

        static final class NV {
            int id;
            String metadata, category;
            int maxLyr;

            NV(int id, String metadata, String category, int maxLyr) {
                this.id = id;
                this.metadata = metadata;
                this.category = category;
                this.maxLyr = maxLyr;
            }
        }

        static final class EV {
            int src, dst, lyr;

            EV(int src, int dst, int lyr) {
                this.src = src;
                this.dst = dst;
                this.lyr = lyr;
            }
        }

        static final class GraphInfo {
            int topLayer, nodeCount;
            int[] nodesPerLayer, edgesPerLayer;
            List<NV> nodes = new ArrayList<>();
            List<EV> edges = new ArrayList<>();
        }

        GraphInfo getInfo() {
            GraphInfo gi = new GraphInfo();
            gi.topLayer = topLayer;
            gi.nodeCount = G.size();
            int maxL = Math.max(topLayer + 1, 1);
            gi.nodesPerLayer = new int[maxL];
            gi.edgesPerLayer = new int[maxL];
            for (int id : G.keySet()) {
                Node nd = G.get(id);
                gi.nodes.add(new NV(id, nd.item.metadata, nd.item.category, nd.maxLyr));
                for (int lc = 0; lc <= nd.maxLyr && lc < maxL; lc++) {
                    gi.nodesPerLayer[lc]++;
                    if (lc < nd.nbrs.size())
                        for (int nid : nd.nbrs.get(lc))
                            if (id < nid) {
                                gi.edgesPerLayer[lc]++;
                                gi.edges.add(new EV(id, nid, lc));
                            }
                }
            }
            return gi;
        }

        int size() {
            return G.size();
        }
    }

    // =====================================================================
    // VECTOR DATABASE (demo 16D index)
    // =====================================================================
    static final class VectorDB {
        private final StdIntMap<VectorItem> store = new StdIntMap<>();
        private final BruteForce bf = new BruteForce();
        private final KDTree kdt;
        private final HNSW hnsw = new HNSW(16, 200);
        private int nextId = 1;
        final int dims;

        VectorDB(int d) {
            this.kdt = new KDTree(d);
            this.dims = d;
        }

        synchronized int insert(String meta, String cat, float[] emb, DistFn dist) {
            VectorItem v = new VectorItem(nextId++, meta, cat, emb);
            store.put(v.id, v);
            bf.insert(v);
            kdt.insert(v);
            hnsw.insert(v, dist);
            return v.id;
        }

        synchronized boolean remove(int id) {
            if (!store.containsKey(id)) return false;
            store.remove(id);
            bf.remove(id);
            hnsw.remove(id);
            kdt.rebuild(new ArrayList<>(store.values()));
            return true;
        }

        static final class Hit {
            int id;
            String meta, cat;
            float[] emb;
            float dist;

            Hit(int id, String meta, String cat, float[] emb, float dist) {
                this.id = id;
                this.meta = meta;
                this.cat = cat;
                this.emb = emb;
                this.dist = dist;
            }
        }

        static final class SearchOut {
            List<Hit> hits = new ArrayList<>();
            long us;
            String algo, metric;
        }

        synchronized SearchOut search(float[] q, int k, String metric, String algo) {
            DistFn dfn = getDistFn(metric);
            long t0 = System.nanoTime();
            List<Pair> raw;
            if (algo.equals("bruteforce")) raw = bf.knn(q, k, dfn);
            else if (algo.equals("kdtree")) raw = kdt.knn(q, k, dfn);
            else raw = hnsw.knn(q, k, 50, dfn);
            long us = (System.nanoTime() - t0) / 1000L;

            SearchOut out = new SearchOut();
            out.us = us;
            out.algo = algo;
            out.metric = metric;
            for (Pair p : raw) {
                VectorItem it = store.get(p.id);
                if (it != null) out.hits.add(new Hit(p.id, it.metadata, it.category, it.emb, p.dist));
            }
            return out;
        }

        static final class BenchOut {
            long bfUs, kdUs, hnswUs;
            int n;
        }

        private static long time(Runnable fn) {
            long t = System.nanoTime();
            fn.run();
            return (System.nanoTime() - t) / 1000L;
        }

        synchronized BenchOut benchmark(float[] q, int k, String metric) {
            DistFn dfn = getDistFn(metric);
            BenchOut b = new BenchOut();
            b.bfUs = time(() -> bf.knn(q, k, dfn));
            b.kdUs = time(() -> kdt.knn(q, k, dfn));
            b.hnswUs = time(() -> hnsw.knn(q, k, 50, dfn));
            b.n = store.size();
            return b;
        }

        synchronized List<VectorItem> all() {
            return new ArrayList<>(store.values());
        }

        synchronized HNSW.GraphInfo hnswInfo() {
            return hnsw.getInfo();
        }

        synchronized int size() {
            return store.size();
        }
    }

    // =====================================================================
    // JSON HELPERS
    // =====================================================================
    static String jS(String s) {
        StringBuilder o = new StringBuilder("\"");
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '"') o.append("\\\"");
            else if (c == '\\') o.append("\\\\");
            else if (c == '\n') o.append("\\n");
            else if (c == '\r') o.append("\\r");
            else if (c == '\t') o.append("\\t");
            else o.append(c);
        }
        return o.append('"').toString();
    }

    static String fmt(double v, int prec) {
        return String.format(Locale.ROOT, "%." + prec + "f", v);
    }

    static String jVec(float[] v) {
        StringBuilder ss = new StringBuilder("[");
        for (int i = 0; i < v.length; i++) {
            if (i > 0) ss.append(',');
            ss.append(fmt(v[i], 4));
        }
        return ss.append(']').toString();
    }

    private static final Pattern FLOAT_PREFIX =
            Pattern.compile("^\\s*[+-]?(?:\\d+\\.?\\d*|\\.\\d+)(?:[eE][+-]?\\d+)?");

    /** std::stof equivalent (parses a numeric prefix, null if none). */
    static Float stof(String t) {
        Matcher m = FLOAT_PREFIX.matcher(t);
        if (!m.find()) return null;
        try {
            return Float.parseFloat(m.group().trim());
        } catch (NumberFormatException e) {
            return null;
        }
    }

    static float[] parseVec(String s) {
        List<Float> v = new ArrayList<>();
        int start = 0;
        while (start <= s.length()) {
            int comma = s.indexOf(',', start);
            String t = (comma < 0) ? s.substring(start) : s.substring(start, comma);
            if (!(comma < 0 && t.isEmpty())) {
                Float f = stof(t);
                if (f != null) v.add(f);
            }
            if (comma < 0) break;
            start = comma + 1;
        }
        float[] r = new float[v.size()];
        for (int i = 0; i < r.length; i++) r[i] = v.get(i);
        return r;
    }

    /** Extract a JSON string field value (handles basic escape sequences) */
    static String extractStr(String body, String key) {
        int p = body.indexOf("\"" + key + "\"");
        if (p < 0) return "";
        int colon = body.indexOf(':', p);
        p = colon + 1;
        while (p < body.length() && (body.charAt(p) == ' ' || body.charAt(p) == '\t')) p++;
        if (p >= body.length() || body.charAt(p) != '"') return "";
        p++;
        StringBuilder result = new StringBuilder();
        while (p < body.length()) {
            char c = body.charAt(p);
            if (c == '"') break;
            if (c == '\\' && p + 1 < body.length()) {
                p++;
                switch (body.charAt(p)) {
                    case '"': result.append('"'); break;
                    case '\\': result.append('\\'); break;
                    case 'n': result.append('\n'); break;
                    case 'r': result.append('\r'); break;
                    case 't': result.append('\t'); break;
                    default: result.append(body.charAt(p)); break;
                }
            } else {
                result.append(c);
            }
            p++;
        }
        return result.toString();
    }

    private static final Pattern INT_PREFIX = Pattern.compile("^\\s*[+-]?\\d+");

    /** Extract a JSON integer field value */
    static int extractInt(String body, String key, int def) {
        int p = body.indexOf("\"" + key + "\"");
        if (p < 0) return def;
        p = body.indexOf(':', p) + 1;
        while (p < body.length() && (body.charAt(p) == ' ' || body.charAt(p) == '\t')) p++;
        Matcher m = INT_PREFIX.matcher(body.substring(Math.min(p, body.length())));
        if (!m.find()) return def;
        try {
            return Integer.parseInt(m.group().trim().replace("+", ""));
        } catch (NumberFormatException e) {
            return def;
        }
    }

    static final class ParsedBody {
        String meta, cat;
        float[] emb;
    }

    static ParsedBody parseBody(String b) {
        ParsedBody pb = new ParsedBody();
        pb.meta = extractStr(b, "metadata");
        pb.cat = extractStr(b, "category");
        pb.emb = new float[0];
        int p = b.indexOf("\"embedding\"");
        if (p >= 0) {
            p = b.indexOf('[', p);
            if (p >= 0) {
                int e = b.indexOf(']', p);
                if (e >= 0) pb.emb = parseVec(b.substring(p + 1, e));
            }
        }
        if (pb.meta.isEmpty() || pb.emb.length == 0) return null;
        return pb;
    }

    // =====================================================================
    // TEXT CHUNKER
    // =====================================================================
    static List<String> chunkText(String text, int chunkWords, int overlapWords) {
        List<String> words = new ArrayList<>();
        StringTokenizer st = new StringTokenizer(text, " \t\n\r\f\u000B");
        while (st.hasMoreTokens()) words.add(st.nextToken());
        if (words.isEmpty()) return new ArrayList<>();
        if (words.size() <= chunkWords) return new ArrayList<>(Collections.singletonList(text));

        List<String> chunks = new ArrayList<>();
        int step = chunkWords - overlapWords;
        for (int i = 0; i < words.size(); i += step) {
            int end = Math.min(i + chunkWords, words.size());
            StringBuilder chunk = new StringBuilder();
            for (int j = i; j < end; j++) {
                if (j > i) chunk.append(' ');
                chunk.append(words.get(j));
            }
            chunks.add(chunk.toString());
            if (end == words.size()) break;
        }
        return chunks;
    }

    // =====================================================================
    // OLLAMA CLIENT — wraps local Ollama REST API
    // Install: https://ollama.com
    // Models:  ollama pull nomic-embed-text
    //          ollama pull llama3.2
    // =====================================================================
    static final class OllamaClient {
        private final String host;
        private final int port;
        String embedModel = "nomic-embed-text";
        String genModel = "llama3.2";

        OllamaClient(String h, int p) {
            this.host = h;
            this.port = p;
        }

        OllamaClient() {
            this("127.0.0.1", 11434);
        }

        private static final class Resp {
            final int status;
            final String body;

            Resp(int status, String body) {
                this.status = status;
                this.body = body;
            }
        }

        private Resp request(String method, String path, String body, int connectSecs, int readSecs) throws IOException {
            HttpURLConnection c = (HttpURLConnection) new URL("http://" + host + ":" + port + path).openConnection();
            try {
                c.setConnectTimeout(connectSecs * 1000);
                c.setReadTimeout(readSecs * 1000);
                c.setUseCaches(false);
                c.setRequestMethod(method);
                if (body != null) {
                    byte[] b = body.getBytes(StandardCharsets.UTF_8);
                    c.setDoOutput(true);
                    c.setRequestProperty("Content-Type", "application/json");
                    c.setFixedLengthStreamingMode(b.length);
                    try (OutputStream os = c.getOutputStream()) {
                        os.write(b);
                    }
                }
                int st = c.getResponseCode();
                InputStream is = (st >= 400) ? c.getErrorStream() : c.getInputStream();
                String text = (is == null) ? "" : new String(readAll(is), StandardCharsets.UTF_8);
                return new Resp(st, text);
            } finally {
                c.disconnect();
            }
        }

        // Escape a string for embedding inside a JSON string literal
        private static String esc(String s) {
            String q = jS(s);
            return q.substring(1, q.length() - 1);
        }

        // Parse {"embedding":[...]} from Ollama /api/embeddings response
        private static float[] parseEmbedding(String body) {
            int p = body.indexOf("\"embedding\"");
            if (p < 0) return new float[0];
            p = body.indexOf('[', p);
            if (p < 0) return new float[0];
            int e = p + 1, depth = 1;
            while (e < body.length() && depth > 0) {
                char c = body.charAt(e);
                if (c == '[') depth++;
                else if (c == ']') depth--;
                e++;
            }
            return parseVec(body.substring(p + 1, e - 1));
        }

        boolean isAvailable() {
            try {
                Resp res = request("GET", "/api/tags", null, 2, 2);
                return res.status == 200;
            } catch (Exception e) {
                return false;
            }
        }

        // Returns empty array if Ollama is not running or model not found
        float[] embed(String text) {
            try {
                String body = "{\"model\":\"" + embedModel + "\",\"prompt\":\"" + esc(text) + "\"}";
                Resp res = request("POST", "/api/embeddings", body, 3, 30);
                if (res.status != 200) return new float[0];
                return parseEmbedding(res.body);
            } catch (Exception e) {
                return new float[0];
            }
        }

        // Returns error string if Ollama is unavailable
        String generate(String prompt) {
            try {
                String body = "{\"model\":\"" + genModel + "\","
                        + "\"prompt\":\"" + esc(prompt) + "\","
                        + "\"stream\":false}";
                Resp res = request("POST", "/api/generate", body, 3, 180); // LLMs can be slow
                if (res.status != 200) return "ERROR: Ollama unavailable. Run: ollama serve";
                return extractStr(res.body, "response");
            } catch (Exception e) {
                return "ERROR: Ollama unavailable. Run: ollama serve";
            }
        }
    }

    // =====================================================================
    // DOCUMENT DATABASE — HNSW over real Ollama embeddings
    // =====================================================================
    static final class DocItem {
        final int id;
        final String title;
        final String text;
        final float[] emb;

        DocItem(int id, String title, String text, float[] emb) {
            this.id = id;
            this.title = title;
            this.text = text;
            this.emb = emb;
        }
    }

    static final class DocHit {
        final float dist;
        final DocItem item;

        DocHit(float dist, DocItem item) {
            this.dist = dist;
            this.item = item;
        }
    }

    static final class DocumentDB {
        private final StdIntMap<DocItem> store = new StdIntMap<>();
        private final HNSW hnsw = new HNSW(16, 200);
        private final BruteForce bf = new BruteForce(); // brute force fallback for small sets
        private int nextId = 1;
        private int dims = 0; // determined from first inserted embedding

        // Insert one chunk with its pre-computed embedding
        synchronized int insert(String title, String text, float[] emb) {
            if (dims == 0) dims = emb.length;
            DocItem item = new DocItem(nextId++, title, text, emb);
            store.put(item.id, item);
            VectorItem vi = new VectorItem(item.id, title, "doc", emb);
            hnsw.insert(vi, Main::cosine);
            bf.insert(vi);
            return item.id;
        }

        // Semantic search — returns top-k most similar chunks
        synchronized List<DocHit> search(float[] q, int k, float maxDist) {
            if (store.isEmpty()) return new ArrayList<>();
            List<Pair> raw = (store.size() < 10)
                    ? bf.knn(q, k, Main::cosine)
                    : hnsw.knn(q, k, 50, Main::cosine);
            List<DocHit> out = new ArrayList<>();
            for (Pair p : raw) {
                DocItem it = store.get(p.id);
                if (it != null && p.dist <= maxDist) out.add(new DocHit(p.dist, it));
            }
            return out;
        }

        synchronized boolean remove(int id) {
            if (!store.containsKey(id)) return false;
            store.remove(id);
            hnsw.remove(id);
            bf.remove(id);
            return true;
        }

        synchronized List<DocItem> all() {
            return new ArrayList<>(store.values());
        }

        synchronized int size() {
            return store.size();
        }

        synchronized int getDims() {
            return dims;
        }
    }

    // =====================================================================
    // DEMO DATA (16D categorical vectors)
    // =====================================================================
    static float[] fv(float... v) {
        return v;
    }

    static void loadDemo(VectorDB db) {
        DistFn dist = getDistFn("cosine");
        // Dims 0-3: CS | Dims 4-7: Math | Dims 8-11: Food | Dims 12-15: Sports

        db.insert("Linked List: nodes connected by pointers", "cs",
                fv(0.90f, 0.85f, 0.72f, 0.68f, 0.12f, 0.08f, 0.15f, 0.10f, 0.05f, 0.08f, 0.06f, 0.09f, 0.07f, 0.11f, 0.08f, 0.06f), dist);
        db.insert("Binary Search Tree: O(log n) search and insert", "cs",
                fv(0.88f, 0.82f, 0.78f, 0.74f, 0.15f, 0.10f, 0.08f, 0.12f, 0.06f, 0.07f, 0.08f, 0.05f, 0.09f, 0.06f, 0.07f, 0.10f), dist);
        db.insert("Dynamic Programming: memoization overlapping subproblems", "cs",
                fv(0.82f, 0.76f, 0.88f, 0.80f, 0.20f, 0.18f, 0.12f, 0.09f, 0.07f, 0.06f, 0.08f, 0.07f, 0.08f, 0.09f, 0.06f, 0.07f), dist);
        db.insert("Graph BFS and DFS: breadth and depth first traversal", "cs",
                fv(0.85f, 0.80f, 0.75f, 0.82f, 0.18f, 0.14f, 0.10f, 0.08f, 0.06f, 0.09f, 0.07f, 0.06f, 0.10f, 0.08f, 0.09f, 0.07f), dist);
        db.insert("Hash Table: O(1) lookup with collision chaining", "cs",
                fv(0.87f, 0.78f, 0.70f, 0.76f, 0.13f, 0.11f, 0.09f, 0.14f, 0.08f, 0.07f, 0.06f, 0.08f, 0.07f, 0.10f, 0.08f, 0.09f), dist);

        db.insert("Calculus: derivatives integrals and limits", "math",
                fv(0.12f, 0.15f, 0.18f, 0.10f, 0.91f, 0.86f, 0.78f, 0.72f, 0.08f, 0.06f, 0.07f, 0.09f, 0.07f, 0.08f, 0.06f, 0.10f), dist);
        db.insert("Linear Algebra: matrices eigenvalues eigenvectors", "math",
                fv(0.20f, 0.18f, 0.15f, 0.12f, 0.88f, 0.90f, 0.82f, 0.76f, 0.09f, 0.07f, 0.08f, 0.06f, 0.10f, 0.07f, 0.08f, 0.09f), dist);
        db.insert("Probability: distributions random variables Bayes theorem", "math",
                fv(0.15f, 0.12f, 0.20f, 0.18f, 0.84f, 0.80f, 0.88f, 0.82f, 0.07f, 0.08f, 0.06f, 0.10f, 0.09f, 0.06f, 0.09f, 0.08f), dist);
        db.insert("Number Theory: primes modular arithmetic RSA cryptography", "math",
                fv(0.22f, 0.16f, 0.14f, 0.20f, 0.80f, 0.85f, 0.76f, 0.90f, 0.08f, 0.09f, 0.07f, 0.06f, 0.08f, 0.10f, 0.07f, 0.06f), dist);
        db.insert("Combinatorics: permutations combinations generating functions", "math",
                fv(0.18f, 0.20f, 0.16f, 0.14f, 0.86f, 0.78f, 0.84f, 0.80f, 0.06f, 0.07f, 0.09f, 0.08f, 0.06f, 0.09f, 0.10f, 0.07f), dist);

        db.insert("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
                fv(0.08f, 0.06f, 0.09f, 0.07f, 0.07f, 0.08f, 0.06f, 0.09f, 0.90f, 0.86f, 0.78f, 0.72f, 0.08f, 0.06f, 0.09f, 0.07f), dist);
        db.insert("Sushi: vinegared rice raw fish and nori rolls", "food",
                fv(0.06f, 0.08f, 0.07f, 0.09f, 0.09f, 0.06f, 0.08f, 0.07f, 0.86f, 0.90f, 0.82f, 0.76f, 0.07f, 0.09f, 0.06f, 0.08f), dist);
        db.insert("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
                fv(0.09f, 0.07f, 0.06f, 0.08f, 0.08f, 0.09f, 0.07f, 0.06f, 0.82f, 0.78f, 0.90f, 0.84f, 0.09f, 0.07f, 0.08f, 0.06f), dist);
        db.insert("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
                fv(0.07f, 0.09f, 0.08f, 0.06f, 0.06f, 0.07f, 0.09f, 0.08f, 0.78f, 0.82f, 0.86f, 0.90f, 0.06f, 0.08f, 0.07f, 0.09f), dist);
        db.insert("Croissant: laminated pastry with buttery flaky layers", "food",
                fv(0.06f, 0.07f, 0.10f, 0.09f, 0.10f, 0.06f, 0.07f, 0.10f, 0.85f, 0.80f, 0.76f, 0.82f, 0.09f, 0.07f, 0.10f, 0.06f), dist);

        db.insert("Basketball: fast-paced shooting dribbling slam dunks", "sports",
                fv(0.09f, 0.07f, 0.08f, 0.10f, 0.08f, 0.09f, 0.07f, 0.06f, 0.08f, 0.07f, 0.09f, 0.06f, 0.91f, 0.85f, 0.78f, 0.72f), dist);
        db.insert("Football: tackles touchdowns field goals and strategy", "sports",
                fv(0.07f, 0.09f, 0.06f, 0.08f, 0.09f, 0.07f, 0.10f, 0.08f, 0.07f, 0.09f, 0.08f, 0.07f, 0.87f, 0.89f, 0.82f, 0.76f), dist);
        db.insert("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
                fv(0.08f, 0.06f, 0.09f, 0.07f, 0.07f, 0.08f, 0.06f, 0.09f, 0.09f, 0.06f, 0.07f, 0.08f, 0.83f, 0.80f, 0.88f, 0.82f), dist);
        db.insert("Chess: openings endgames tactics strategic board game", "sports",
                fv(0.25f, 0.20f, 0.22f, 0.18f, 0.22f, 0.18f, 0.20f, 0.15f, 0.06f, 0.08f, 0.07f, 0.09f, 0.80f, 0.84f, 0.78f, 0.90f), dist);
        db.insert("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
                fv(0.06f, 0.08f, 0.07f, 0.09f, 0.08f, 0.06f, 0.09f, 0.07f, 0.10f, 0.08f, 0.06f, 0.07f, 0.85f, 0.82f, 0.86f, 0.80f), dist);
    }

    // =====================================================================
    // HTTP SERVER HELPERS
    // =====================================================================
    static void cors(HttpExchange ex) {
        ex.getResponseHeaders().set("Access-Control-Allow-Origin", "*");
        ex.getResponseHeaders().set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
        ex.getResponseHeaders().set("Access-Control-Allow-Headers", "Content-Type");
    }

    static void send(HttpExchange ex, int status, String contentType, String body) throws IOException {
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().set("Content-Type", contentType);
        ex.sendResponseHeaders(status, bytes.length);
        try (OutputStream os = ex.getResponseBody()) {
            os.write(bytes);
        }
    }

    static void json(HttpExchange ex, String body) throws IOException {
        send(ex, 200, "application/json", body);
    }

    static byte[] readAll(InputStream in) throws IOException {
        ByteArrayOutputStream bo = new ByteArrayOutputStream();
        byte[] buf = new byte[8192];
        int n;
        while ((n = in.read(buf)) != -1) bo.write(buf, 0, n);
        return bo.toByteArray();
    }

    static String urlDecode(String s) {
        try {
            return URLDecoder.decode(s, "UTF-8");
        } catch (UnsupportedEncodingException e) {
            return s;
        }
    }

    static Map<String, String> queryParams(URI uri) {
        Map<String, String> m = new HashMap<>();
        String raw = uri.getRawQuery();
        if (raw == null || raw.isEmpty()) return m;
        for (String kv : raw.split("&")) {
            int eq = kv.indexOf('=');
            String k = eq < 0 ? kv : kv.substring(0, eq);
            String v = eq < 0 ? "" : kv.substring(eq + 1);
            m.put(urlDecode(k), urlDecode(v));
        }
        return m;
    }

    static int stoiOr(String s, int def) {
        Matcher m = INT_PREFIX.matcher(s);
        if (!m.find()) return def;
        try {
            return Integer.parseInt(m.group().trim().replace("+", ""));
        } catch (NumberFormatException e) {
            return def;
        }
    }

    static String readIndexHtml() {
        Path[] candidates = {Paths.get("index.html"), Paths.get("..", "index.html")};
        for (Path p : candidates) {
            try {
                if (Files.isRegularFile(p)) return new String(Files.readAllBytes(p), StandardCharsets.UTF_8);
            } catch (IOException ignored) {
            }
        }
        return null;
    }

    static final Pattern DELETE_RE = Pattern.compile("^/delete/(\\d+)$");
    static final Pattern DOC_DELETE_RE = Pattern.compile("^/doc/delete/(\\d+)$");

    // =====================================================================
    // MAIN
    // =====================================================================
    public static void main(String[] args) throws Exception {
        final VectorDB db = new VectorDB(DIMS);
        final DocumentDB docDB = new DocumentDB();
        final OllamaClient ollama = new OllamaClient();

        loadDemo(db);

        // Check Ollama at startup (non-fatal)
        boolean ollamaUp = ollama.isAvailable();

        System.out.println("=== VectorDB Engine ===");
        System.out.println("http://localhost:8080");
        System.out.println(db.size() + " demo vectors | " + DIMS + " dims | HNSW+KD-Tree+BruteForce");
        System.out.println("Ollama: " + (ollamaUp ? "ONLINE" : "OFFLINE (install from ollama.com)"));
        if (ollamaUp)
            System.out.println("  embed model: " + ollama.embedModel + "  gen model: " + ollama.genModel);

        HttpServer svr = HttpServer.create(new InetSocketAddress(8080), 0);

        svr.createContext("/", ex -> {
            try {
                handle(ex, db, docDB, ollama);
            } catch (Throwable t) {
                try {
                    send(ex, 500, "text/plain", "500 Internal Server Error");
                } catch (Throwable ignored) {
                }
            } finally {
                ex.close();
            }
        });

        svr.setExecutor(Executors.newCachedThreadPool());
        svr.start();
    }

    static void handle(HttpExchange ex, VectorDB db, DocumentDB docDB, OllamaClient ollama) throws IOException {
        final String method = ex.getRequestMethod();
        final String path = ex.getRequestURI().getPath();
        final Map<String, String> params = queryParams(ex.getRequestURI());

        // CORS preflight
        if (method.equals("OPTIONS")) {
            cors(ex);
            ex.sendResponseHeaders(204, -1);
            return;
        }

        // Serve index.html
        if (method.equals("GET") && path.equals("/")) {
            String html = readIndexHtml();
            if (html == null) {
                ex.sendResponseHeaders(404, -1);
            } else {
                send(ex, 200, "text/html", html);
            }
            return;
        }

        // ── DEMO VECTOR ENDPOINTS ─────────────────────────────────────

        if (method.equals("GET") && path.equals("/search")) {
            cors(ex);
            float[] q = parseVec(params.getOrDefault("v", ""));
            if (q.length != DIMS) {
                json(ex, "{\"error\":\"need " + DIMS + "D vector\"}");
                return;
            }
            int k = stoiOr(params.getOrDefault("k", ""), 5);
            String metric = params.getOrDefault("metric", "");
            if (metric.isEmpty()) metric = "cosine";
            String algo = params.getOrDefault("algo", "");
            if (algo.isEmpty()) algo = "hnsw";

            VectorDB.SearchOut out = db.search(q, k, metric, algo);

            StringBuilder ss = new StringBuilder();
            ss.append("{\"results\":[");
            for (int i = 0; i < out.hits.size(); i++) {
                if (i > 0) ss.append(',');
                VectorDB.Hit h = out.hits.get(i);
                ss.append("{\"id\":").append(h.id)
                        .append(",\"metadata\":").append(jS(h.meta))
                        .append(",\"category\":").append(jS(h.cat))
                        .append(",\"distance\":").append(fmt(h.dist, 6))
                        .append(",\"embedding\":").append(jVec(h.emb)).append('}');
            }
            ss.append("],\"latencyUs\":").append(out.us)
                    .append(",\"algo\":").append(jS(out.algo))
                    .append(",\"metric\":").append(jS(out.metric)).append('}');
            json(ex, ss.toString());
            return;
        }

        if (method.equals("POST") && path.equals("/insert")) {
            cors(ex);
            String body = new String(readAll(ex.getRequestBody()), StandardCharsets.UTF_8);
            ParsedBody pb = parseBody(body);
            if (pb == null || pb.emb.length != DIMS) {
                json(ex, "{\"error\":\"invalid body\"}");
                return;
            }
            int id = db.insert(pb.meta, pb.cat, pb.emb, getDistFn("cosine"));
            json(ex, "{\"id\":" + id + "}");
            return;
        }

        if (method.equals("DELETE")) {
            Matcher m = DELETE_RE.matcher(path);
            if (m.matches()) {
                cors(ex);
                int id = Integer.parseInt(m.group(1));
                boolean ok = db.remove(id);
                json(ex, "{\"ok\":" + (ok ? "true" : "false") + "}");
                return;
            }
        }

        if (method.equals("GET") && path.equals("/items")) {
            cors(ex);
            List<VectorItem> items = db.all();
            StringBuilder ss = new StringBuilder("[");
            for (int i = 0; i < items.size(); i++) {
                if (i > 0) ss.append(',');
                VectorItem v = items.get(i);
                ss.append("{\"id\":").append(v.id)
                        .append(",\"metadata\":").append(jS(v.metadata))
                        .append(",\"category\":").append(jS(v.category))
                        .append(",\"embedding\":").append(jVec(v.emb)).append('}');
            }
            ss.append(']');
            json(ex, ss.toString());
            return;
        }

        if (method.equals("GET") && path.equals("/benchmark")) {
            cors(ex);
            float[] q = parseVec(params.getOrDefault("v", ""));
            if (q.length != DIMS) {
                json(ex, "{\"error\":\"need " + DIMS + "D vector\"}");
                return;
            }
            int k = stoiOr(params.getOrDefault("k", ""), 5);
            String metric = params.getOrDefault("metric", "");
            if (metric.isEmpty()) metric = "cosine";
            VectorDB.BenchOut b = db.benchmark(q, k, metric);
            json(ex, "{\"bruteforceUs\":" + b.bfUs + ",\"kdtreeUs\":" + b.kdUs
                    + ",\"hnswUs\":" + b.hnswUs + ",\"itemCount\":" + b.n + "}");
            return;
        }

        if (method.equals("GET") && path.equals("/hnsw-info")) {
            cors(ex);
            HNSW.GraphInfo gi = db.hnswInfo();
            StringBuilder ss = new StringBuilder();
            ss.append("{\"topLayer\":").append(gi.topLayer).append(",\"nodeCount\":").append(gi.nodeCount)
                    .append(",\"nodesPerLayer\":[");
            for (int i = 0; i < gi.nodesPerLayer.length; i++) {
                if (i > 0) ss.append(',');
                ss.append(gi.nodesPerLayer[i]);
            }
            ss.append("],\"edgesPerLayer\":[");
            for (int i = 0; i < gi.edgesPerLayer.length; i++) {
                if (i > 0) ss.append(',');
                ss.append(gi.edgesPerLayer[i]);
            }
            ss.append("],\"nodes\":[");
            for (int i = 0; i < gi.nodes.size(); i++) {
                if (i > 0) ss.append(',');
                HNSW.NV n = gi.nodes.get(i);
                ss.append("{\"id\":").append(n.id).append(",\"metadata\":").append(jS(n.metadata))
                        .append(",\"category\":").append(jS(n.category)).append(",\"maxLyr\":").append(n.maxLyr).append('}');
            }
            ss.append("],\"edges\":[");
            for (int i = 0; i < gi.edges.size(); i++) {
                if (i > 0) ss.append(',');
                HNSW.EV e = gi.edges.get(i);
                ss.append("{\"src\":").append(e.src).append(",\"dst\":").append(e.dst).append(",\"lyr\":").append(e.lyr).append('}');
            }
            ss.append("]}");
            json(ex, ss.toString());
            return;
        }

        // ── DOCUMENT + RAG ENDPOINTS ──────────────────────────────────

        // POST /doc/insert {"title":"...","text":"..."}
        // Chunks the text, embeds each chunk via Ollama, stores in DocumentDB
        if (method.equals("POST") && path.equals("/doc/insert")) {
            cors(ex);
            String body = new String(readAll(ex.getRequestBody()), StandardCharsets.UTF_8);
            String title = extractStr(body, "title");
            String text = extractStr(body, "text");
            if (title.isEmpty() || text.isEmpty()) {
                json(ex, "{\"error\":\"need title and text\"}");
                return;
            }

            List<String> chunks = chunkText(text, 250, 30);
            List<Integer> ids = new ArrayList<>();
            for (int i = 0; i < chunks.size(); i++) {
                float[] emb = ollama.embed(chunks.get(i));
                if (emb.length == 0) {
                    json(ex, "{\"error\":\"Ollama unavailable. "
                            + "Install from https://ollama.com then run: "
                            + "ollama pull nomic-embed-text && ollama pull llama3.2\"}");
                    return;
                }
                String chunkTitle = (chunks.size() > 1)
                        ? title + " [" + (i + 1) + "/" + chunks.size() + "]"
                        : title;
                ids.add(docDB.insert(chunkTitle, chunks.get(i), emb));
            }

            StringBuilder ss = new StringBuilder("{\"ids\":[");
            for (int i = 0; i < ids.size(); i++) {
                if (i > 0) ss.append(',');
                ss.append(ids.get(i));
            }
            ss.append("],\"chunks\":").append(chunks.size())
                    .append(",\"dims\":").append(docDB.getDims()).append('}');
            json(ex, ss.toString());
            return;
        }

        // DELETE /doc/delete/123
        if (method.equals("DELETE")) {
            Matcher m = DOC_DELETE_RE.matcher(path);
            if (m.matches()) {
                cors(ex);
                int id = Integer.parseInt(m.group(1));
                boolean ok = docDB.remove(id);
                json(ex, "{\"ok\":" + (ok ? "true" : "false") + "}");
                return;
            }
        }

        // GET /doc/list
        if (method.equals("GET") && path.equals("/doc/list")) {
            cors(ex);
            List<DocItem> docs = docDB.all();
            StringBuilder ss = new StringBuilder("[");
            for (int i = 0; i < docs.size(); i++) {
                if (i > 0) ss.append(',');
                DocItem d = docs.get(i);
                // Truncate text preview to 120 chars
                String preview = d.text.substring(0, Math.min(120, d.text.length()));
                if (d.text.length() > 120) preview += "\u2026";
                int spaces = 0;
                for (int c = 0; c < d.text.length(); c++) if (d.text.charAt(c) == ' ') spaces++;
                ss.append("{\"id\":").append(d.id)
                        .append(",\"title\":").append(jS(d.title))
                        .append(",\"preview\":").append(jS(preview))
                        .append(",\"words\":").append(spaces + 1)
                        .append('}');
            }
            ss.append(']');
            json(ex, ss.toString());
            return;
        }

        // POST /doc/search {"question":"...","k":3}
        // Fast retrieval for the UI visualizer
        if (method.equals("POST") && path.equals("/doc/search")) {
            cors(ex);
            String body = new String(readAll(ex.getRequestBody()), StandardCharsets.UTF_8);
            String question = extractStr(body, "question");
            int k = extractInt(body, "k", 3);
            if (question.isEmpty()) {
                json(ex, "{\"error\":\"need question\"}");
                return;
            }
            float[] qEmb = ollama.embed(question);
            if (qEmb.length == 0) {
                json(ex, "{\"error\":\"Ollama unavailable\"}");
                return;
            }
            List<DocHit> hits = docDB.search(qEmb, k, 0.7f);
            StringBuilder ss = new StringBuilder("{\"contexts\":[");
            for (int i = 0; i < hits.size(); i++) {
                if (i > 0) ss.append(',');
                ss.append("{\"id\":").append(hits.get(i).item.id)
                        .append(",\"title\":").append(jS(hits.get(i).item.title))
                        .append(",\"distance\":").append(fmt(hits.get(i).dist, 4)).append('}');
            }
            ss.append("]}");
            json(ex, ss.toString());
            return;
        }

        // POST /doc/ask {"question":"...","k":3}
        // Full RAG pipeline: embed -> retrieve -> generate
        if (method.equals("POST") && path.equals("/doc/ask")) {
            cors(ex);
            String body = new String(readAll(ex.getRequestBody()), StandardCharsets.UTF_8);
            String question = extractStr(body, "question");
            int k = extractInt(body, "k", 3);
            if (question.isEmpty()) {
                json(ex, "{\"error\":\"need question\"}");
                return;
            }

            // Step 1: embed the question
            float[] qEmb = ollama.embed(question);
            if (qEmb.length == 0) {
                json(ex, "{\"error\":\"Ollama unavailable\"}");
                return;
            }

            // Step 2: retrieve top-k relevant chunks
            List<DocHit> hits = docDB.search(qEmb, k, 0.7f);

            // Step 3: build prompt
            StringBuilder ctx = new StringBuilder();
            for (int i = 0; i < hits.size(); i++) {
                ctx.append("[").append(i + 1).append("] ").append(hits.get(i).item.title).append(":\n")
                        .append(hits.get(i).item.text).append("\n\n");
            }
            String prompt =
                    "You are a helpful assistant. Answer the user's question directly. "
                    + "Use the provided context if it contains relevant information. "
                    + "If it doesn't, just use your own general knowledge. "
                    + "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like 'the context doesn't mention'. "
                    + "Just answer the question naturally.\n\n"
                    + "Context:\n" + ctx
                    + "Question: " + question + "\n\n"
                    + "Answer:";

            // Step 4: generate answer
            String answer = ollama.generate(prompt);

            // Step 5: return everything
            StringBuilder ss = new StringBuilder();
            ss.append("{\"answer\":").append(jS(answer))
                    .append(",\"model\":").append(jS(ollama.genModel))
                    .append(",\"contexts\":[");
            for (int i = 0; i < hits.size(); i++) {
                if (i > 0) ss.append(',');
                DocItem d = hits.get(i).item;
                ss.append("{\"id\":").append(d.id)
                        .append(",\"title\":").append(jS(d.title))
                        .append(",\"text\":").append(jS(d.text))
                        .append(",\"distance\":").append(fmt(hits.get(i).dist, 4)).append('}');
            }
            ss.append("],\"docCount\":").append(docDB.size()).append('}');
            json(ex, ss.toString());
            return;
        }

        // GET /status
        if (method.equals("GET") && path.equals("/status")) {
            cors(ex);
            boolean up = ollama.isAvailable();
            json(ex, "{\"ollamaAvailable\":" + (up ? "true" : "false")
                    + ",\"embedModel\":" + jS(ollama.embedModel)
                    + ",\"genModel\":" + jS(ollama.genModel)
                    + ",\"docCount\":" + docDB.size()
                    + ",\"docDims\":" + docDB.getDims()
                    + ",\"demoDims\":" + DIMS
                    + ",\"demoCount\":" + db.size() + "}");
            return;
        }

        if (method.equals("GET") && path.equals("/stats")) {
            cors(ex);
            json(ex, "{\"count\":" + db.size()
                    + ",\"dims\":" + DIMS
                    + ",\"algorithms\":[\"bruteforce\",\"kdtree\",\"hnsw\"]"
                    + ",\"metrics\":[\"euclidean\",\"cosine\",\"manhattan\"]}");
            return;
        }

        ex.sendResponseHeaders(404, -1);
    }
}
