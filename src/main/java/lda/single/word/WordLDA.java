package lda.single.word;

/**
 * Created by leleyu on 2016/9/4.
 */
public interface WordLDA {
  void init();
  void trainOneIteration(int iter);
}
