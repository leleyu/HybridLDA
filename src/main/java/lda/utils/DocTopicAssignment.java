package lda.utils;

import java.util.Arrays;

/*! This class stores the doc-topic assignment */
public class DocTopicAssignment {

  private final static int INIT_SIZE = 2;
  private final static int INC_SIZE  = 1;

  /*! total topics number, equals K */
  private final int topicNum;

  /*! topic indices */
  private int[] topics;
  /*! topic counts */
  private int[] counts;
  /*! element number */
  private int size;

  public DocTopicAssignment(int topicNum) {
    this(topicNum, INIT_SIZE);
  }

  public DocTopicAssignment(int topicNum, int initSize) {
    this.topicNum = topicNum;
    this.size = 0;
    this.topics = new int[initSize];
    this.counts = new int[initSize];
  }

  public void writeToDense(int[] dense) {
    Arrays.fill(dense, 0);
    for (int k = 0; k < size; k ++) {
      dense[topics[k]] = counts[k];
    }
  }

  public void writeToDense(int[] dense, int[] docTopics, int[] topicIndexMap) {
    Arrays.fill(dense, 0);
    System.arraycopy(topics, 0, docTopics, 0, size);
    for (int k = 0; k < size; k ++) {
      dense[topics[k]] = counts[k];
      topicIndexMap[topics[k]] = k;
    }
  }

  public void readFromDense(int[] dense, int[] docTopics, int length) {
    size = length;
    if (size > topics.length) {
      topics = new int[size];
      counts = new int[size];
    }

    for (int j = 0; j < size; j ++) {
      topics[j] = docTopics[j];
      counts[j] = dense[docTopics[j]];
    }
  }

  public void readFromDense(int[] dense) {
    int elementNum = 0;
    for (int k = 0; k < topicNum; k ++) {
      if (dense[k] != 0)
        elementNum ++;
    }

    size = elementNum;
    if (size > topics.length) {
      int allocSize = Math.min(topicNum, size + INC_SIZE);
      topics = new int[allocSize];
      counts = new int[allocSize];
    }

    int index = 0;
    for (int k = 0; k < topicNum; k ++) {
      if (dense[k] != 0) {
        topics[index] = k;
        counts[index] = dense[k];
        index ++;
      }
    }
  }


  public int size() {
    return size;
  }

  public int getTopic(int index) {
    return topics[index];
  }

  public int getTopicCount(int index) {
    return counts[index];
  }
}
