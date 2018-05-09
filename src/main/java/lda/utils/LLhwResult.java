package lda.utils;

/**
 * Created by leleyu on 2016/3/9.
 */
public class LLhwResult {

  public double llhwSum;
  public int tokenNum;
  public int docTopicNNZ;

  public LLhwResult() {
    llhwSum = 0.0;
    tokenNum = 0;
    docTopicNNZ = 0;
  }

  public void plusBy(LLhwResult other) {
    llhwSum += other.llhwSum;
    tokenNum += other.tokenNum;
    docTopicNNZ += other.docTopicNNZ;
  }

  public double value() {
    return llhwSum / tokenNum;
  }
}
