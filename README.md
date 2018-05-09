# Single Machine version of LDA*

## Compile
```sh
cd jni
make 
make jni

cd ..
mvn package
```

## Run Hybrid LDA example
```sh
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



