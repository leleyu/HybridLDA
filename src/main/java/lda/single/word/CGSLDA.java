package lda.single.word;

import lda.utils.BinarySearch;
import lda.utils.Document;
import lda.utils.TokensAndParams;
import lda.utils.Utils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import java.io.IOException;
import java.util.List;
import java.util.Random;

/**
 * Created by leleyu on 2016/9/4.
 */
public class CGSLDA implements WordLDA {

  private final static Log LOG = LogFactory.getLog(CGSLDA.class);

  public int D, V, K;
  public double alpha, beta, vbeta;

  int[][] ndk;
  int[]   nk;
  double[] p;

  List<Document> docs;
  TokensAndParams[] words;

  public CGSLDA(int D, int V, int K, double alpha, double beta,
                List<Document> docs) {
    this.D = D;
    this.V = V;
    this.K = K;
    this.alpha = alpha;
    this.beta = beta;
    this.vbeta = V * beta;
    this.docs = docs;
  }

  @Override
  public void init() {
    nk = new int[K];
    ndk = new int[D][K];
    p = new double[K];

    for (int d = 0; d < D; d ++)
      ndk[d] = new int[K];

    words = Utils.buildTokensByWord(docs, K, V);

    int did, tt;
    Random rand = new Random(System.currentTimeMillis());
    for (TokensAndParams param: words) {
      if (param == null)
        continue;

      for (int d = 0; d < param.tokens.length; d ++) {
        did = param.tokens[d];
        tt  = rand.nextInt(K);
        param.wtrow[tt] ++;
        nk[tt] ++;
        ndk[did][tt] ++;
        param.topics[d] = (short) tt;
      }
    }
  }

  @Override
  public void trainOneIteration(int iter) {
    long start, end;

    start = System.currentTimeMillis();
    for (TokensAndParams param: words)
      if (param != null)
        sampleOneWord(param);

    end = System.currentTimeMillis();

    long train_tt = end - start;

    start = System.currentTimeMillis();
    double llhw = new Utils().loglilikehood2(alpha, beta, words, ndk, nk, K, V, docs);
    end = System.currentTimeMillis();

    long eval_tt = end - start;

    LOG.info(String.format("iter=%d train_tt=%d eval_tt=%d llhw=%f",
            iter, train_tt, eval_tt, llhw));
  }

  public void testOneIteration() {

  }

  public void sampleOneWord(TokensAndParams param) {
    int[] wa = param.wtrow;
    int did, tt, ttt;

    double psum = 0.0;

    Random rand = new Random(System.currentTimeMillis());

    for (int d = 0; d < param.tokens.length; d ++) {
      did = param.tokens[d];
      tt  = param.topics[d];

      wa[tt] --;
      nk[tt] --;
      ndk[did][tt] --;

      psum = 0.0;
      for (int k = 0; k < K; k ++) {
        psum += (ndk[did][k] + alpha) * (wa[k] + beta) / (nk[k] + vbeta);
        p[k] = psum;
      }

      double u = rand.nextDouble() * psum;
      ttt = BinarySearch.binarySearch(p, u, 0, K - 1);

      param.topics[d] = (short) ttt;
      wa[ttt] ++;
      nk[ttt] ++;
      ndk[did][ttt] ++;
    }
  }

  public static void nips(String[] argv) throws IOException {
    String path = "nips.train";
    List<Document> docs = Utils.read(path);
    int D = docs.size();
    int V = 12420;
    int K = 1024;
    double alpha = 0.1;
    double beta  = 0.1;

    CGSLDA lda = new CGSLDA(D, V, K, alpha, beta, docs);
    lda.init();

    for (int i = 0; i < 200; i ++) {
      lda.trainOneIteration(i);
      lda.testOneIteration();
    }
  }

  public static void main(String[] argv) throws IOException {
    nips(argv);
  }
}
