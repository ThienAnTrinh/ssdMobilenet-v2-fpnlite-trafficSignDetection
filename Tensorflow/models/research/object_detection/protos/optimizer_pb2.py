# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: object_detection/protos/optimizer.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\'object_detection/protos/optimizer.proto\x12\x17object_detection.protos\"\xb5\x02\n\tOptimizer\x12G\n\x12rms_prop_optimizer\x18\x01 \x01(\x0b\x32).object_detection.protos.RMSPropOptimizerH\x00\x12H\n\x12momentum_optimizer\x18\x02 \x01(\x0b\x32*.object_detection.protos.MomentumOptimizerH\x00\x12@\n\x0e\x61\x64\x61m_optimizer\x18\x03 \x01(\x0b\x32&.object_detection.protos.AdamOptimizerH\x00\x12 \n\x12use_moving_average\x18\x04 \x01(\x08:\x04true\x12$\n\x14moving_average_decay\x18\x05 \x01(\x02:\x06\x30.9999B\x0b\n\toptimizer\"\x9f\x01\n\x10RMSPropOptimizer\x12<\n\rlearning_rate\x18\x01 \x01(\x0b\x32%.object_detection.protos.LearningRate\x12%\n\x18momentum_optimizer_value\x18\x02 \x01(\x02:\x03\x30.9\x12\x12\n\x05\x64\x65\x63\x61y\x18\x03 \x01(\x02:\x03\x30.9\x12\x12\n\x07\x65psilon\x18\x04 \x01(\x02:\x01\x31\"x\n\x11MomentumOptimizer\x12<\n\rlearning_rate\x18\x01 \x01(\x0b\x32%.object_detection.protos.LearningRate\x12%\n\x18momentum_optimizer_value\x18\x02 \x01(\x02:\x03\x30.9\"e\n\rAdamOptimizer\x12<\n\rlearning_rate\x18\x01 \x01(\x0b\x32%.object_detection.protos.LearningRate\x12\x16\n\x07\x65psilon\x18\x02 \x01(\x02:\x05\x31\x65-08\"\x80\x03\n\x0cLearningRate\x12O\n\x16\x63onstant_learning_rate\x18\x01 \x01(\x0b\x32-.object_detection.protos.ConstantLearningRateH\x00\x12`\n\x1f\x65xponential_decay_learning_rate\x18\x02 \x01(\x0b\x32\x35.object_detection.protos.ExponentialDecayLearningRateH\x00\x12T\n\x19manual_step_learning_rate\x18\x03 \x01(\x0b\x32/.object_detection.protos.ManualStepLearningRateH\x00\x12V\n\x1a\x63osine_decay_learning_rate\x18\x04 \x01(\x0b\x32\x30.object_detection.protos.CosineDecayLearningRateH\x00\x42\x0f\n\rlearning_rate\"4\n\x14\x43onstantLearningRate\x12\x1c\n\rlearning_rate\x18\x01 \x01(\x02:\x05\x30.002\"\xef\x01\n\x1c\x45xponentialDecayLearningRate\x12$\n\x15initial_learning_rate\x18\x01 \x01(\x02:\x05\x30.002\x12\x1c\n\x0b\x64\x65\x63\x61y_steps\x18\x02 \x01(\r:\x07\x34\x30\x30\x30\x30\x30\x30\x12\x1a\n\x0c\x64\x65\x63\x61y_factor\x18\x03 \x01(\x02:\x04\x30.95\x12\x17\n\tstaircase\x18\x04 \x01(\x08:\x04true\x12\x1f\n\x14\x62urnin_learning_rate\x18\x05 \x01(\x02:\x01\x30\x12\x17\n\x0c\x62urnin_steps\x18\x06 \x01(\r:\x01\x30\x12\x1c\n\x11min_learning_rate\x18\x07 \x01(\x02:\x01\x30\"\xf1\x01\n\x16ManualStepLearningRate\x12$\n\x15initial_learning_rate\x18\x01 \x01(\x02:\x05\x30.002\x12V\n\x08schedule\x18\x02 \x03(\x0b\x32\x44.object_detection.protos.ManualStepLearningRate.LearningRateSchedule\x12\x15\n\x06warmup\x18\x03 \x01(\x08:\x05\x66\x61lse\x1a\x42\n\x14LearningRateSchedule\x12\x0c\n\x04step\x18\x01 \x01(\r\x12\x1c\n\rlearning_rate\x18\x02 \x01(\x02:\x05\x30.002\"\xbe\x01\n\x17\x43osineDecayLearningRate\x12!\n\x12learning_rate_base\x18\x01 \x01(\x02:\x05\x30.002\x12\x1c\n\x0btotal_steps\x18\x02 \x01(\r:\x07\x34\x30\x30\x30\x30\x30\x30\x12$\n\x14warmup_learning_rate\x18\x03 \x01(\x02:\x06\x30.0002\x12\x1b\n\x0cwarmup_steps\x18\x04 \x01(\r:\x05\x31\x30\x30\x30\x30\x12\x1f\n\x14hold_base_rate_steps\x18\x05 \x01(\r:\x01\x30')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'object_detection.protos.optimizer_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _globals['_OPTIMIZER']._serialized_start=69
  _globals['_OPTIMIZER']._serialized_end=378
  _globals['_RMSPROPOPTIMIZER']._serialized_start=381
  _globals['_RMSPROPOPTIMIZER']._serialized_end=540
  _globals['_MOMENTUMOPTIMIZER']._serialized_start=542
  _globals['_MOMENTUMOPTIMIZER']._serialized_end=662
  _globals['_ADAMOPTIMIZER']._serialized_start=664
  _globals['_ADAMOPTIMIZER']._serialized_end=765
  _globals['_LEARNINGRATE']._serialized_start=768
  _globals['_LEARNINGRATE']._serialized_end=1152
  _globals['_CONSTANTLEARNINGRATE']._serialized_start=1154
  _globals['_CONSTANTLEARNINGRATE']._serialized_end=1206
  _globals['_EXPONENTIALDECAYLEARNINGRATE']._serialized_start=1209
  _globals['_EXPONENTIALDECAYLEARNINGRATE']._serialized_end=1448
  _globals['_MANUALSTEPLEARNINGRATE']._serialized_start=1451
  _globals['_MANUALSTEPLEARNINGRATE']._serialized_end=1692
  _globals['_MANUALSTEPLEARNINGRATE_LEARNINGRATESCHEDULE']._serialized_start=1626
  _globals['_MANUALSTEPLEARNINGRATE_LEARNINGRATESCHEDULE']._serialized_end=1692
  _globals['_COSINEDECAYLEARNINGRATE']._serialized_start=1695
  _globals['_COSINEDECAYLEARNINGRATE']._serialized_end=1885
# @@protoc_insertion_point(module_scope)
