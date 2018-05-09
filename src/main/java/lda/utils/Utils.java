package lda.utils;

import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.shorts.ShortArrayList;
import lda.sparse.S2BTightTraverseMap;
import lda.sparse.S2SSparseMap;
import lda.sparse.S2STightTraverseMap;
import lda.sparse.TraverseHashMap;
import org.apache.commons.math3.special.Gamma;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

/**
 * Created by leleyu on 2016/8/18.
 */
public class Utils {

  public static List<Document> read(String path) throws IOException {
    BufferedReader reader = new BufferedReader(new FileReader(new File(path)));
    List<Document> docs = new ArrayList<>();
    String line;

    int docId = 0;
    while ((line = reader.readLine()) != null) {
      String[] splits = line.split(" ");
      int length = splits.length;
      int[] wordIds = new int[length];
      for (int i = 0; i < length; i++) {
        wordIds[i] = Integer.parseInt(splits[i]);
      }

      Document document = new Document(docId, wordIds);
      docs.add(document);
      docId++;
    }

    return docs;
  }

  public static List<Document> read2(String path) throws IOException {
    BufferedReader reader = new BufferedReader(new FileReader(new File(path)));
    List<Document> docs   = new ArrayList<>();
    String line;
    String[] parts;

    int D = Integer.parseInt(reader.readLine());
    int V = Integer.parseInt(reader.readLine());
    int N = Integer.parseInt(reader.readLine());

//    Document[] docs = new Document[D];

    Map<Integer, IntArrayList> tdocs = new HashMap<>();
    int did, wid, cnt;
    for (int i = 0; i < N; i ++) {
      line = reader.readLine();
      parts = line.split(" ");
      did   = Integer.parseInt(parts[0]) - 1;
      wid   = Integer.parseInt(parts[1]) - 1;
      cnt   = Integer.parseInt(parts[2]);

      if (tdocs.get(did) == null) {
        tdocs.put(did, new IntArrayList());
      }

      for (int j = 0; j < cnt; j ++) {
        tdocs.get(did).add(wid);
      }
    }

    for (did = 0; did < D; did ++) {
      if (tdocs.containsKey(did)) {
        Document doc = new Document(did, tdocs.get(did).toIntArray());
        docs.add(doc);
      }
    }

    reader.close();

    return docs;
  }

  public static int readV(String path) throws IOException {
    BufferedReader reader = new BufferedReader(new FileReader(new File(path)));

    int D = Integer.parseInt(reader.readLine());
    int V = Integer.parseInt(reader.readLine());
    return V;
  }

//  public static BlockManager buildAndSerializeTokens(List<Document> docs, int K) {
//    Map<Integer, IntArrayList> builder = new HashMap<>();
//
//    int totalDocs = docs.size();
//    for (int docId = 0; docId < totalDocs; docId++) {
//      Document doc = docs.get(docId);
//      int size = doc.getLength();
//      for (int i = 0; i < size; i++) {
//        int wordId = doc.getWordId(i);
//        if (!builder.containsKey(wordId)) {
//          builder.put(wordId, new IntArrayList());
//        }
//        builder.get(wordId).add(docId);
//      }
//      doc.reset();
//    }
//
//    Iterator<Map.Entry<Integer, IntArrayList>> iter;
//    iter = builder.entrySet().iterator();
//    BlockManager manager = new BlockManager();
//    while (iter.hasNext()) {
//      Map.Entry<Integer, IntArrayList> entry = iter.next();
//      TokensAndParams one = new TokensAndParams(entry.getKey(), entry.getValue().toIntArray(), K);
//      manager.write(one);
//    }
//
//    manager.writeFinish();
//    return manager;
//  }

  public static TokensAndParams[] buildTokensByWord(List<Document> docs, int K, int V) {
    Map<Integer, IntArrayList> builder = new HashMap<>();

    int D = docs.size();
    Document doc;
    int wid;
    for (int d = 0; d < D; d ++) {
      doc = docs.get(d);
      for (int w = 0; w < doc.wids.length; w ++) {
        wid = doc.wids[w];
        if (!builder.containsKey(wid)) {
          builder.put(wid, new IntArrayList());
        }
        builder.get(wid).add(d);
      }

      doc.reset();
    }

    Iterator<Map.Entry<Integer, IntArrayList>> iter;
    iter = builder.entrySet().iterator();
    TokensAndParams[] params = new TokensAndParams[V];
    while (iter.hasNext()) {
      Map.Entry<Integer, IntArrayList> entry = iter.next();
      TokensAndParams param = new TokensAndParams(entry.getKey(), entry.getValue().toIntArray(),
              K);
      params[entry.getKey()] = param;
    }

    return params;
  }

  public static TokensAndParams[] buildTokensByWordForLightLDA(List<Document> docs, int K, int V) {
    Map<Integer, IntArrayList> builder = new HashMap<>();
    Map<Integer, ShortArrayList> builder1 = new HashMap<>();

    int D = docs.size();
    Document doc;
    int wid;
    for (int d = 0; d < D; d ++) {
      doc = docs.get(d);
      for (short w = 0; w < doc.wids.length; w ++) {
        wid = doc.wids[w];
        if (!builder.containsKey(wid)) {
          builder.put(wid, new IntArrayList());
          builder1.put(wid, new ShortArrayList());
        }
        builder.get(wid).add(d);
        builder1.get(wid).add(w);
      }

      doc.wids = null;
    }

    Iterator<Map.Entry<Integer, IntArrayList>> iter;
    iter = builder.entrySet().iterator();
    TokensAndParams[] params = new TokensAndParams[V];

    while (iter.hasNext()) {
      Map.Entry<Integer, IntArrayList> entry = iter.next();
      TokensAndParams param = new TokensAndParams(entry.getKey(), entry.getValue().toIntArray(), K);
      ShortArrayList idx = builder1.get(entry.getKey());
      for (int i = 0; i < param.topics.length; i ++)
        param.topics[i] = idx.getShort(i);
      params[entry.getKey()] = param;
    }

    return params;
  }

  public static int[] randamArray(int len) {
    Random rand = new Random(System.currentTimeMillis());
    int[] arr = new int[len];
    for (int i = 0; i < len; i++)
      arr[i] = rand.nextInt();
    return arr;
  }


  public LLhwResult loglikelihood(double alpha, double beta,
                                  int K, int V,
                                  int[][] n_dk, int[][] n_wk, int[] n_k,
                                  List<Document> docs) {
    LLhwResult result = computeDocLLH(alpha, K, n_dk, docs);
    System.out.println("doc llhw " + result.llhwSum);
    double wordllhw = computeWordLLH(0, V, beta, n_wk, K);
    System.out.println("word llhw " + wordllhw);
    result.llhwSum += wordllhw;
    double sumllhw = computeWordLLHSummary(K, V, beta, n_k);
    System.out.println("word summary llhw " + sumllhw);
    result.llhwSum += sumllhw;
    return result;
  }

  public LLhwResult computeDocLLH(double alpha, int K, int[][] n_dk, List<Document> docs) {
    LLhwResult result = new LLhwResult();

    double alphaSum = alpha * K;
    for (Document doc : docs) {
      computeDocLLH(result, doc, n_dk, alpha, K);
    }

    result.llhwSum -= result.docTopicNNZ * Gamma.logGamma(alpha);
    result.llhwSum += docs.size() * Gamma.logGamma(alphaSum);
    return result;
  }


  public void computeDocLLH(LLhwResult result, Document doc, int[][] n_dk, double alpha, int K) {
    int[] da = n_dk[doc.docId];
    if (da != null) {
      result.tokenNum += doc.getLength();
      int nnz = 0;
      for (int j = 0; j < da.length; j++) {
        if (da[j] != 0) {
          result.llhwSum += Gamma.logGamma(alpha + da[j]);
          nnz++;
        }
      }
      result.docTopicNNZ += nnz;
      result.llhwSum -= Gamma.logGamma(doc.getLength() + alpha * K);
    }
  }

  public double computeWordLLH(int swid, int ewid, double beta, int[][] n_wk, int K) {

    double ll = 0.0;
    double lgammaBeta = Gamma.logGamma(beta);
    for (int wid = swid; wid < ewid; wid++) {
      int[] wa = n_wk[wid];
      for (int k = 0; k < K; k++)
        if (wa[k] > 0) {
          ll += Gamma.logGamma(wa[k] + beta) - lgammaBeta;
        }
    }
    return ll;
  }

  public double computeWordLLHSummary(int K, int V, double beta, int[] n_k) {
    double ll = 0.0;
    ll += K * Gamma.logGamma(beta * V);
    for (int k = 0; k < K; k++) {
      ll -= Gamma.logGamma(n_k[k] + beta * V);
    }
    return ll;
  }


  public double perplexity(List<Document> docs, int[][] n_dk, int[][] n_wk, int K) {
    double perplexity = 0.0;
    int D = docs.size();
    Document doc;
    int tokenNum = 0;
    for (int d = 0; d < D; d++) {
      doc = docs.get(d);
      int did = doc.docId;
      tokenNum += doc.wids.length;
      for (int w = 0; w < doc.wids.length; w++) {
        int wid = doc.wids[w];
        double tt = 0.0;
        for (int k = 0; k < K; k++) {
          tt += n_dk[did][k] * n_wk[wid][k];
        }

        perplexity += Math.log(tt);
      }
    }
    perplexity = Math.exp(-perplexity / tokenNum);
    return perplexity;
  }

  public double loglilikehood2(double alpha, double beta, TokensAndParams[] params, S2STightTraverseMap[] ndk,
                               int[] nk, int K, int V, List<Document> docs) {

    double lgamma_beta, lgamma_alpha;

    lgamma_alpha = Gamma.logGamma(alpha);
    lgamma_beta  = Gamma.logGamma(beta);

    double ll = 0.0;

    ll += K * Gamma.logGamma(beta * V);

    for (int k = 0; k < K; k++) {
      ll -= Gamma.logGamma(beta * V + nk[k]);
    }

    for (int w = 0; w < V; w++) {
      if (params[w] == null)
        continue;
      int[] wa = params[w].wtrow;
      for (int k = 0; k < K; k++) {
        if (wa[k] > 0) {
          ll += Gamma.logGamma(beta + wa[k]) - lgamma_beta;
        }
      }
    }

    int D = docs.size();
    for (int d = 0; d < D; d++) {
      Document doc = docs.get(d);
      S2STightTraverseMap da = ndk[d];
      ll += (Gamma.logGamma(alpha * K) - Gamma.logGamma(alpha * K + doc.length));
      for (int i = 0; i < da.size; i ++) {
        ll += Gamma.logGamma(alpha + da.getVal(i)) - lgamma_alpha;
      }
    }

    return ll;
  }

  public double loglilikehood2(double alpha, double beta, TokensAndParams[] params, S2SSparseMap[] ndk,
                               int[] nk, int K, int V, List<Document> docs) {

    double lgamma_beta, lgamma_alpha;

    lgamma_alpha = Gamma.logGamma(alpha);
    lgamma_beta  = Gamma.logGamma(beta);

    double ll = 0.0;

    ll += K * Gamma.logGamma(beta * V);

    for (int k = 0; k < K; k++) {
      ll -= Gamma.logGamma(beta * V + nk[k]);
    }

    for (int w = 0; w < V; w++) {
      if (params[w] == null)
        continue;
      int[] wa = params[w].wtrow;
      for (int k = 0; k < K; k++) {
        if (wa[k] > 0) {
          ll += Gamma.logGamma(beta + wa[k]) - lgamma_beta;
        }
      }
    }

    int D = docs.size();
    for (int d = 0; d < D; d++) {
      Document doc = docs.get(d);
      S2SSparseMap da = ndk[d];
      ll += (Gamma.logGamma(alpha * K) - Gamma.logGamma(alpha * K + doc.length));
      for (int i = 0; i < da.n; i ++) {
        if (da.key[i] != -1)
          ll += Gamma.logGamma(alpha + da.value[i]) - lgamma_alpha;
      }
    }

    return ll;
  }

  public double loglilikehood2(double alpha, double beta, TokensAndParams[] params, TraverseHashMap[] ndk,
                               int[] nk, int K, int V, List<Document> docs) {

    double lgamma_beta, lgamma_alpha;

    lgamma_alpha = Gamma.logGamma(alpha);
    lgamma_beta  = Gamma.logGamma(beta);

    double ll = 0.0;

    ll += K * Gamma.logGamma(beta * V);

    for (int k = 0; k < K; k++) {
      ll -= Gamma.logGamma(beta * V + nk[k]);
    }

    for (int w = 0; w < V; w++) {
      if (params[w] == null)
        continue;
      int[] wa = params[w].wtrow;
      for (int k = 0; k < K; k++) {
        if (wa[k] > 0) {
          ll += Gamma.logGamma(beta + wa[k]) - lgamma_beta;
        }
      }
    }

    int D = docs.size();
    for (int d = 0; d < D; d++) {
      Document doc = docs.get(d);

      TraverseHashMap da = ndk[d];
      ll += (Gamma.logGamma(alpha * K) - Gamma.logGamma(alpha * K + doc.length));
      for (int i = 0; i < da.size; i++) {
        ll += Gamma.logGamma(alpha + da.getVal(i)) - lgamma_alpha;
      }
    }

    return ll;
  }

  public double loglilikehood2(double alpha, double beta, TokensAndParams[] params, TraverseMap[] ndk,
                               int[] nk, int K, int V, List<Document> docs) {

    double lgamma_beta, lgamma_alpha;

    lgamma_alpha = Gamma.logGamma(alpha);
    lgamma_beta  = Gamma.logGamma(beta);

    double ll = 0.0;

    ll += K * Gamma.logGamma(beta * V);

    for (int k = 0; k < K; k++) {
      ll -= Gamma.logGamma(beta * V + nk[k]);
    }

    for (int w = 0; w < V; w++) {
      if (params[w] == null)
        continue;
      int[] wa = params[w].wtrow;
      for (int k = 0; k < K; k++) {
        if (wa[k] > 0) {
          ll += Gamma.logGamma(beta + wa[k]) - lgamma_beta;
        }
      }
    }

    int D = docs.size();
    for (int d = 0; d < D; d++) {
      Document doc = docs.get(d);
      TraverseMap da = ndk[d];
      ll += (Gamma.logGamma(alpha * K) - Gamma.logGamma(alpha * K + doc.length));
      for (int i = 0; i < da.size; i ++) {
        ll += Gamma.logGamma(alpha + da.value[da.idx[i]]) - lgamma_alpha;
      }
    }

    return ll;
  }

  public double docllhw(TraverseMap[] ndk, int[] docLens, int D, double alpha, int K) {

    double lgamma_alpha = Gamma.logGamma(alpha);
    double ll = 0.0;
    for (int d = 0; d < D; d++) {
      TraverseMap da = ndk[d];
      ll += (Gamma.logGamma(alpha * K) - Gamma.logGamma(alpha * K + docLens[d]));
      for (int i = 0; i < da.size; i ++) {
        ll += Gamma.logGamma(alpha + da.value[da.idx[i]]) - lgamma_alpha;
      }
    }
    return ll;
  }

  public double nkllhw(int[] nk, double beta, int V, int K) {
    double ll = 0.0;
    ll += K * Gamma.logGamma(beta * V);

    for (int k = 0; k < K; k++) {
      ll -= Gamma.logGamma(beta * V + nk[k]);
    }
    return ll;
  }

  public double loglilikehood2(double alpha, double beta, TokensAndParams[] params, int[][] ndk,
                               int[] nk, int K, int V, List<Document> docs) {

    double lgamma_beta, lgamma_alpha;

    lgamma_alpha = Gamma.logGamma(alpha);
    lgamma_beta  = Gamma.logGamma(beta);

    double ll = 0.0;

    ll += K * Gamma.logGamma(beta * V);

    for (int k = 0; k < K; k++) {
      ll -= Gamma.logGamma(beta * V + nk[k]);
    }

    for (int w = 0; w < V; w++) {
      if (params[w] == null)
        continue;
      int[] wa = params[w].wtrow;
      for (int k = 0; k < K; k++) {
        if (wa[k] > 0) {
          ll += Gamma.logGamma(beta + wa[k]) - lgamma_beta;
        }
      }
    }

    int D = docs.size();
    for (int d = 0; d < D; d++) {
      Document doc = docs.get(d);
      int[] da = ndk[d];
      ll += (Gamma.logGamma(alpha * K) - Gamma.logGamma(alpha * K + doc.length));
      for (int k = 0; k < K; k ++) {
        if (da[k] > 0)
          ll += Gamma.logGamma(alpha + da[k]) - lgamma_alpha;
      }
    }

    return ll;
  }

  public double loglilikehood2(double alpha, double beta, int[][] nwk, SparseDocTopic[] ndk,
                               int[] nk, int K, int V, List<Document> docs) {

    double lgamma_beta, lgamma_alpha;

    lgamma_alpha = Gamma.logGamma(alpha);
    lgamma_beta  = Gamma.logGamma(beta);

    double ll = 0.0;

    ll += K * Gamma.logGamma(beta * V);

    for (int k = 0; k < K; k++) {
      ll -= Gamma.logGamma(beta * V + nk[k]);
    }

    for (int w = 0; w < V; w++) {
      int[] wa = nwk[w];
      for (int k = 0; k < K; k++) {
        if (wa[k] > 0) {
          ll += Gamma.logGamma(beta + wa[k]) - lgamma_beta;
        }
      }
    }

    int D = docs.size();
    for (int d = 0; d < D; d++) {
      Document doc = docs.get(d);

      SparseDocTopic da = ndk[d];
      short[] values = da.getValue();
      boolean[] used = da.getUsed();
      ll += (Gamma.logGamma(alpha * K) - Gamma.logGamma(alpha * K + doc.length));
      for (int i = 0; i < used.length; i ++) {
        if (used[i]) {
          ll += Gamma.logGamma(alpha + values[i]) - lgamma_alpha;
        }
      }
    }

    return ll;
  }

  public double loglilikehood2(double alpha, double beta, int[][] nwk, S2STightTraverseMap[] ndk,
                                int[] nk, int K, int V, List<Document> docs) {

    double lgamma_beta, lgamma_alpha;

    lgamma_alpha = Gamma.logGamma(alpha);
    lgamma_beta  = Gamma.logGamma(beta);

    double ll = 0.0;

    ll += K * Gamma.logGamma(beta * V);

    for (int k = 0; k < K; k++) {
      ll -= Gamma.logGamma(beta * V + nk[k]);
    }

    for (int w = 0; w < V; w++) {
      int[] wa = nwk[w];
      for (int k = 0; k < K; k++) {
        if (wa[k] > 0) {
          ll += Gamma.logGamma(beta + wa[k]) - lgamma_beta;
        }
      }
    }

    int D = docs.size();
    for (int d = 0; d < D; d++) {
      Document doc = docs.get(d);
      S2STightTraverseMap da = ndk[d];
      ll += (Gamma.logGamma(alpha * K) - Gamma.logGamma(alpha * K + doc.length));
      for (int i = 0; i < da.size; i ++) {
        ll += Gamma.logGamma(alpha + da.getVal(i)) - lgamma_alpha;
      }
    }

    return ll;
  }

  public double loglilikehood2(double alpha, double beta, int[][] nwk, int[][] ndk,
                               int[] nk, int K, int V, List<Document> docs) {

    double lgamma_beta, lgamma_alpha;

    lgamma_beta = Gamma.logGamma(beta);
    lgamma_alpha = Gamma.logGamma(alpha);

    double ll = 0.0;

    ll += K * Gamma.logGamma(beta * V);

    for (int k = 0; k < K; k++) {
      ll -= Gamma.logGamma(beta * V + nk[k]);
    }

    for (int w = 0; w < V; w++) {
      for (int k = 0; k < K; k++) {
        if (nwk[w][k] > 0) {
          ll += Gamma.logGamma(beta + nwk[w][k]) - lgamma_beta;
        }
      }
    }

    int D = docs.size();
    for (int d = 0; d < D; d++) {
      Document doc = docs.get(d);
      ll += (Gamma.logGamma(alpha * K) - Gamma.logGamma(alpha * K + doc.wids.length));
      for (int k = 0; k < K; k++) {
        if (ndk[d][k] > 0)
          ll += Gamma.logGamma(alpha + ndk[d][k]) - lgamma_alpha;
      }
    }

    return ll;
  }

  public double loglilikehood(double alpha, double beta, int[][] nwk, int[][] ndk,
                               int[] nk, int K, int V, List<int[]> docs) {

    double lgamma_beta, lgamma_alpha;

    lgamma_beta = Gamma.logGamma(beta);
    lgamma_alpha = Gamma.logGamma(alpha);

    double ll = 0.0;

    ll += K * Gamma.logGamma(beta * V);

    for (int k = 0; k < K; k++) {
      ll -= Gamma.logGamma(beta * V + nk[k]);
    }

    for (int w = 0; w < V; w++) {
      for (int k = 0; k < K; k++) {
        if (nwk[w][k] > 0) {
          ll += Gamma.logGamma(beta + nwk[w][k]) - lgamma_beta;
        }
      }
    }

    int D = docs.size();
    for (int d = 0; d < D; d++) {
      int[] doc = docs.get(d);
      ll += (Gamma.logGamma(alpha * K) - Gamma.logGamma(alpha * K + doc.length));
      for (int k = 0; k < K; k++) {
        if (ndk[d][k] > 0)
          ll += Gamma.logGamma(alpha + ndk[d][k]) - lgamma_alpha;
      }
    }

    return ll;
  }

  public void showTopic(List<Document> docs) {
    int D = docs.size();
    Document doc;
    for (int d = 0; d < D; d++) {
      doc = docs.get(d);
      for (int t = 0; t < doc.topics.length; t++) {
        System.out.print(doc.topics[t] + ", ");
      }
      System.out.println();
    }
  }

  public void check(int[][] nwk, int[][] ndk, int[] nk, List<Document> docs,
                    int K, int V) {
    int D = docs.size();
    Document doc;

    int[][] tnwk = new int[V][];
    for (int i = 0; i < V; i ++)
      tnwk[i] = new int[K];

    int[][] tndk = new int[D][];
    for (int i = 0; i < D; i ++)
      tndk[i] = new int[K];

    int[] tnk = new int[nk.length];

    for (int d = 0; d < D; d ++) {
      doc = docs.get(d);
      for (int i = 0; i < doc.topics.length; i ++) {
        tnk[doc.topics[i]]++;
        tnwk[doc.wids[i]][doc.topics[i]] ++;
        tndk[doc.docId][doc.topics[i]] ++;
      }
    }

    // check nk
    for (int i = 0; i < nk.length; i ++) {
      if (nk[i] != tnk[i]) {
        System.out.format("Error, nk[%d]=%d while tnk[%d]=%d", i, nk[i], i, tnk[i]);
      }
    }


    // check nwk
    for (int w = 0; w < V; w ++) {
      for (int k = 0; k < K; k ++) {
        if (nwk[w][k] != tnwk[w][k]) {
          System.out.format("Error, nwk[%d][%d]=%d while tnwk[%d][%d]=%d", w, k, nwk[w][k],
                  w, k, tnwk[w][k]);
        }
      }
    }

    // check ndk
    for (int d = 0; d < D; d ++) {
      for (int k = 0; k < K; k ++) {
        if (ndk[d][k] != tndk[d][k]) {
          System.out.format("Error, ndk[%d][%d]=%d while tndk[%d][%d]=%d", d, k, ndk[d][k],
                  d, k, tndk[d][k]);
        }
      }
    }
  }

  public static CsrMat buildMat(List<Document> docs, int V) {
    Document doc;
    int[] wcnt = new int[V];
    int[] ws = new int[V + 1];

    int N = 0;
    int D = docs.size();
    // count word
    for (int d = 0; d < D; d ++) {
      doc = docs.get(d);
      for (int w = 0; w < doc.wids.length; w ++)
        wcnt[doc.wids[w]] ++;

      N += doc.length;
    }

    int[] cols = new int[N];

    // build word start idx
    ws[0] = 0;
    for (int i = 0; i < V; i ++) {
      ws[i + 1] = ws[i] + wcnt[i];
    }

    for (int d = D - 1; d >= 0; d --) {
      doc = docs.get(d);
      int wid;
      for (int w = 0; w < doc.wids.length; w ++) {
        wid = doc.wids[w];
        int pos = ws[wid] + (--wcnt[wid]);
        cols[pos] = d;
      }
    }

    return new CsrMat(V, D, ws, cols);
  }
}
