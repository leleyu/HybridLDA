# Single Machine version of LDA*

## Compile
```bash
cd jni
make 
make jni

cd ..
mvn package
```

## Run Hybrid LDA example
```bash
input=data/nips.train
K=1000
V=12420
threadNum=8
iterations=100
split=600
jvmOpts="-Djava.library.path=jni"
java -cp target/hybridlda-1.0-jar-with-dependencies.jar -Xmx1g $jvmOpts lda.parallel.CombineLDA \
	$input $V $K $threadNum $iterations $split
```

where K is the number of topics, V is the number of word, split is the split point for document length.
Documents with length less than split will be sampled by F+LDA while others will be sampled by WarpLDA.



