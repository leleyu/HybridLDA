package lda.utils;

/**
 * Created by leleyu on 2016/3/8.
 */
public class DenseDocTopic {

  private int topicNum;

  private short[] counts;
  private short[] topics;
  private short[] topicIndexMap;
  private short size;

  public DenseDocTopic(int topicNum) {
    this.topicNum = topicNum;
    counts = new short[topicNum];
    topics = new short[topicNum];
    topicIndexMap = new short[topicNum];
    size = 0;
  }

  public void inc(int topic) {
    short shortTopic = (short) topic;
    counts[topic] ++;
    if (counts[topic] == 1) {
      topics[size] = shortTopic;
      topicIndexMap[topic] = size;
      size ++;
    }
  }

  public void dec(int topic) {
    counts[topic] --;
    if (counts[topic] == 0) {
      short index = topicIndexMap[topic];
      topics[index] = topics[size - 1];
      topicIndexMap[topics[index]] = index;
      size --;
    }
  }

  public short size() {
    return size;
  }

  public short getTopic(int index) {
    return topics[index];
  }

  public short getCount(int topic) {
    return counts[topic];
  }

  public short getCountWithIndex(int index) {
    return counts[topics[index]];
  }
}
