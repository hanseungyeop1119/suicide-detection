  *	o??ڸA2x
AIterator::Root::BatchV2::Prefetch::Shuffle::Zip[0]::ParallelMapV2$??EB[?N@!{?q???B@)??EB[?N@1{?q???B@:Preprocessing2N
Iterator::Root::BatchV2?2???X@!??|;n?N@)?#ӡ??I@1?T!z@@:Preprocessing2X
!Iterator::Root::BatchV2::Prefetch ?A_z??G@!ۛ???m=@)?A_z??G@1ۛ???m=@:Preprocessing2f
/Iterator::Root::BatchV2::Prefetch::Shuffle::Zip$<?l??N@!??+έ#C@)0???"??1eJ?)??:Preprocessing2x
AIterator::Root::BatchV2::Prefetch::Shuffle::Zip[1]::ParallelMapV2$??J?????!?7?+$??)??J?????1?7?+$??:Preprocessing2?
NIterator::Root::BatchV2::Prefetch::Shuffle::Zip[1]::ParallelMapV2::TensorSlice$üǙ&l??!UAui#???)üǙ&l??1UAui#???:Preprocessing2a
*Iterator::Root::BatchV2::Prefetch::Shuffle$S?r/0?N@!=O??+C@)o,(?4??1??k8??:Preprocessing2?
NIterator::Root::BatchV2::Prefetch::Shuffle::Zip[0]::ParallelMapV2::TensorSlice$?N?j???!??"?X??)?N?j???1??"?X??:Preprocessing2E
Iterator::Root
??.??X@!m????N@)p\?M4??1aw?9P??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.