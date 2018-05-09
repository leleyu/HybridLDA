package lda.utils;


import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

/**
 * Created by leleyu on 2016/8/18.
 */
public class TokensAndParams {

  public int[] tokens;
  public short[] topics;
  public int[] wtrow;
  public int wid;

  public TokensAndParams() {

  }

  public TokensAndParams(int wid, int[] tokens, int K) {
    this.wid = wid;
    this.tokens = tokens;
    this.topics = new short[tokens.length];
    this.wtrow = new int[K];
  }

//  @Override
//  public void write(DataOutputStream out) throws IOException {
//    int len = tokens.length;
//
//    out.writeInt(wid);
//    out.writeInt(len);
//
//    for (int i = 0; i < len; i ++)
//      out.writeInt(tokens[i]);
//
//    for (int i = 0; i < len; i ++)
//      out.writeShort(topics[i]);
//
//    out.writeInt(wtrow.length);
//    for (int i = 0; i < wtrow.length; i ++)
//      out.writeInt(wtrow[i]);
//  }
//
//  @Override
//  public void read(DataInputStream in) throws IOException {
//    wid = in.readInt();
//    int len = in.readInt();
//
//    tokens  = new int[len];
//    topics  = new short[len];
//
//    for (int i = 0; i < len; i ++)
//      tokens[i] = in.readInt();
//
//    for (int i = 0; i < len; i ++)
//      topics[i] = in.readShort();
//
//    len = in.readInt();
//
//    wtrow = new int[len];
//    for (int i = 0; i < len; i ++)
//      wtrow[i] = in.readInt();
//  }
}
