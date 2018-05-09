package lda.utils;

import it.unimi.dsi.fastutil.shorts.Short2ShortOpenHashMap;

/**
 * Sparse representation for doc-topic distribution
 * Created by leleyu on 2016/6/20.
 */
public class SparseDocTopic extends Short2ShortOpenHashMap {

  private int topicNum;

  public SparseDocTopic(int topicNum, int size) {
    super(size, 1.0F);
    this.topicNum = topicNum;
  }

  public void inc(int topic) {
    addTo((short) topic, (short) 1);
  }

  public void dec(int topic) {
    if (get(topic) == 1) {
      remove((short) topic);
    } else {
      addTo((short) topic, (short) -1);
    }
  }

  public short get(int topic) {
    return get((short) topic);
  }

  public short[] getKey() {
    return key;
  }

  public short[] getValue() {
    return value;
  }

  public boolean[] getUsed() { return used; }

}
