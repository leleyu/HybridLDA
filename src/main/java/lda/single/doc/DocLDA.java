package lda.single.doc;

/**
 * Created by yulele on 16/9/2.
 */
public interface DocLDA {
  void init();
  void trainOneIteration(int iter);
}
