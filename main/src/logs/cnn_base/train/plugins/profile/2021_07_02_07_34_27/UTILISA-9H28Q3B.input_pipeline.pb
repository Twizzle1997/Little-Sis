	??T???b@??T???b@!??T???b@	?K?+?Ӥ??K?+?Ӥ?!?K?+?Ӥ?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??T???b@lxz?,C??AC?i?q?b@Y?Zd;??*	?????j?@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator??j+??%@!?-?o?X@)??j+??%@1?-?o?X@:Preprocessing2F
Iterator::Model?MbX9??!??O???)???߾??1??{????:Preprocessing2P
Iterator::Model::Prefetch??@??ǈ?!j?@:????)??@??ǈ?1j?@:????:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap??y?)?%@!?1?2??X@)"??u??q?1??>???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?K?+?Ӥ?#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	lxz?,C??lxz?,C??!lxz?,C??      ??!       "      ??!       *      ??!       2	C?i?q?b@C?i?q?b@!C?i?q?b@:      ??!       B      ??!       J	?Zd;???Zd;??!?Zd;??R      ??!       Z	?Zd;???Zd;??!?Zd;??JCPU_ONLYY?K?+?Ӥ?b 