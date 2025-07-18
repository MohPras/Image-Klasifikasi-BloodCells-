��
�#�#
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

,
Cos
x"T
y"T"
Ttype:

2
$
DisableCopyOnRead
resource�
A
EnsureShape

input"T
output"T"
shapeshape"	
Ttype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorMod
x"T
y"T
z"T"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
�
ImageProjectiveTransformV3
images"dtype

transforms
output_shape

fill_value
transformed_images"dtype"
dtypetype:
	2	"
interpolationstring"
	fill_modestring
CONSTANT
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
�
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	"
grad_abool( "
grad_bbool( 
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
0
Neg
x"T
y"T"
Ttype:
2
	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
n
	ReverseV2
tensor"T
axis"Tidx
output"T"
Tidxtype0:
2	"
Ttype:
2	

l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
,
Sin
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
^
StatelessRandomGetKeyCounter
seed"Tseed
key
counter"
Tseedtype0	:
2	
�
StatelessRandomUniformV2
shape"Tshape
key
counter
alg
output"dtype"
dtypetype0:
2"
Tshapetype0:
2	
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
,
Tan
x"T
y"T"
Ttype:

2
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
9
VarIsInitializedOp
resource
is_initialized
�"serve*2.18.02v2.18.0-rc2-4-g6550e4bd8028��
�
%seed_generator_5/seed_generator_stateVarHandleOp*
_output_shapes
: *6

debug_name(&seed_generator_5/seed_generator_state/*
dtype0	*
shape:*6
shared_name'%seed_generator_5/seed_generator_state
�
9seed_generator_5/seed_generator_state/Read/ReadVariableOpReadVariableOp%seed_generator_5/seed_generator_state*
_output_shapes
:*
dtype0	
�
%seed_generator_4/seed_generator_stateVarHandleOp*
_output_shapes
: *6

debug_name(&seed_generator_4/seed_generator_state/*
dtype0	*
shape:*6
shared_name'%seed_generator_4/seed_generator_state
�
9seed_generator_4/seed_generator_state/Read/ReadVariableOpReadVariableOp%seed_generator_4/seed_generator_state*
_output_shapes
:*
dtype0	
�
Model_2/dense_2/kernelVarHandleOp*
_output_shapes
: *'

debug_nameModel_2/dense_2/kernel/*
dtype0*
shape:
��*'
shared_nameModel_2/dense_2/kernel
�
*Model_2/dense_2/kernel/Read/ReadVariableOpReadVariableOpModel_2/dense_2/kernel* 
_output_shapes
:
��*
dtype0
�
Model_2/conv2d_4/kernelVarHandleOp*
_output_shapes
: *(

debug_nameModel_2/conv2d_4/kernel/*
dtype0*
shape: *(
shared_nameModel_2/conv2d_4/kernel
�
+Model_2/conv2d_4/kernel/Read/ReadVariableOpReadVariableOpModel_2/conv2d_4/kernel*&
_output_shapes
: *
dtype0
�
Model_2/conv2d_2/biasVarHandleOp*
_output_shapes
: *&

debug_nameModel_2/conv2d_2/bias/*
dtype0*
shape:*&
shared_nameModel_2/conv2d_2/bias
{
)Model_2/conv2d_2/bias/Read/ReadVariableOpReadVariableOpModel_2/conv2d_2/bias*
_output_shapes
:*
dtype0
�
Model_2/dense_2/biasVarHandleOp*
_output_shapes
: *%

debug_nameModel_2/dense_2/bias/*
dtype0*
shape:*%
shared_nameModel_2/dense_2/bias
y
(Model_2/dense_2/bias/Read/ReadVariableOpReadVariableOpModel_2/dense_2/bias*
_output_shapes
:*
dtype0
�
Model_2/conv2d_4/biasVarHandleOp*
_output_shapes
: *&

debug_nameModel_2/conv2d_4/bias/*
dtype0*
shape: *&
shared_nameModel_2/conv2d_4/bias
{
)Model_2/conv2d_4/bias/Read/ReadVariableOpReadVariableOpModel_2/conv2d_4/bias*
_output_shapes
: *
dtype0
�
Model_2/conv2d_3/biasVarHandleOp*
_output_shapes
: *&

debug_nameModel_2/conv2d_3/bias/*
dtype0*
shape:*&
shared_nameModel_2/conv2d_3/bias
{
)Model_2/conv2d_3/bias/Read/ReadVariableOpReadVariableOpModel_2/conv2d_3/bias*
_output_shapes
:*
dtype0
�
Model_2/conv2d_3/kernelVarHandleOp*
_output_shapes
: *(

debug_nameModel_2/conv2d_3/kernel/*
dtype0*
shape:*(
shared_nameModel_2/conv2d_3/kernel
�
+Model_2/conv2d_3/kernel/Read/ReadVariableOpReadVariableOpModel_2/conv2d_3/kernel*&
_output_shapes
:*
dtype0
�
Model_2/conv2d_2/kernelVarHandleOp*
_output_shapes
: *(

debug_nameModel_2/conv2d_2/kernel/*
dtype0*
shape:*(
shared_nameModel_2/conv2d_2/kernel
�
+Model_2/conv2d_2/kernel/Read/ReadVariableOpReadVariableOpModel_2/conv2d_2/kernel*&
_output_shapes
:*
dtype0
�
Model_2/dense_2/bias_1VarHandleOp*
_output_shapes
: *'

debug_nameModel_2/dense_2/bias_1/*
dtype0*
shape:*'
shared_nameModel_2/dense_2/bias_1
}
*Model_2/dense_2/bias_1/Read/ReadVariableOpReadVariableOpModel_2/dense_2/bias_1*
_output_shapes
:*
dtype0
�
#Variable/Initializer/ReadVariableOpReadVariableOpModel_2/dense_2/bias_1*
_class
loc:@Variable*
_output_shapes
:*
dtype0
�
VariableVarHandleOp*
_class
loc:@Variable*
_output_shapes
: *

debug_name	Variable/*
dtype0*
shape:*
shared_name
Variable
a
)Variable/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable*
_output_shapes
: 
_
Variable/AssignAssignVariableOpVariable#Variable/Initializer/ReadVariableOp*
dtype0
a
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
:*
dtype0
�
Model_2/dense_2/kernel_1VarHandleOp*
_output_shapes
: *)

debug_nameModel_2/dense_2/kernel_1/*
dtype0*
shape:
��*)
shared_nameModel_2/dense_2/kernel_1
�
,Model_2/dense_2/kernel_1/Read/ReadVariableOpReadVariableOpModel_2/dense_2/kernel_1* 
_output_shapes
:
��*
dtype0
�
%Variable_1/Initializer/ReadVariableOpReadVariableOpModel_2/dense_2/kernel_1*
_class
loc:@Variable_1* 
_output_shapes
:
��*
dtype0
�

Variable_1VarHandleOp*
_class
loc:@Variable_1*
_output_shapes
: *

debug_nameVariable_1/*
dtype0*
shape:
��*
shared_name
Variable_1
e
+Variable_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_1*
_output_shapes
: 
e
Variable_1/AssignAssignVariableOp
Variable_1%Variable_1/Initializer/ReadVariableOp*
dtype0
k
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1* 
_output_shapes
:
��*
dtype0
�
Model_2/conv2d_4/bias_1VarHandleOp*
_output_shapes
: *(

debug_nameModel_2/conv2d_4/bias_1/*
dtype0*
shape: *(
shared_nameModel_2/conv2d_4/bias_1

+Model_2/conv2d_4/bias_1/Read/ReadVariableOpReadVariableOpModel_2/conv2d_4/bias_1*
_output_shapes
: *
dtype0
�
%Variable_2/Initializer/ReadVariableOpReadVariableOpModel_2/conv2d_4/bias_1*
_class
loc:@Variable_2*
_output_shapes
: *
dtype0
�

Variable_2VarHandleOp*
_class
loc:@Variable_2*
_output_shapes
: *

debug_nameVariable_2/*
dtype0*
shape: *
shared_name
Variable_2
e
+Variable_2/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_2*
_output_shapes
: 
e
Variable_2/AssignAssignVariableOp
Variable_2%Variable_2/Initializer/ReadVariableOp*
dtype0
e
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes
: *
dtype0
�
Model_2/conv2d_4/kernel_1VarHandleOp*
_output_shapes
: **

debug_nameModel_2/conv2d_4/kernel_1/*
dtype0*
shape: **
shared_nameModel_2/conv2d_4/kernel_1
�
-Model_2/conv2d_4/kernel_1/Read/ReadVariableOpReadVariableOpModel_2/conv2d_4/kernel_1*&
_output_shapes
: *
dtype0
�
%Variable_3/Initializer/ReadVariableOpReadVariableOpModel_2/conv2d_4/kernel_1*
_class
loc:@Variable_3*&
_output_shapes
: *
dtype0
�

Variable_3VarHandleOp*
_class
loc:@Variable_3*
_output_shapes
: *

debug_nameVariable_3/*
dtype0*
shape: *
shared_name
Variable_3
e
+Variable_3/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_3*
_output_shapes
: 
e
Variable_3/AssignAssignVariableOp
Variable_3%Variable_3/Initializer/ReadVariableOp*
dtype0
q
Variable_3/Read/ReadVariableOpReadVariableOp
Variable_3*&
_output_shapes
: *
dtype0
�
Model_2/conv2d_3/bias_1VarHandleOp*
_output_shapes
: *(

debug_nameModel_2/conv2d_3/bias_1/*
dtype0*
shape:*(
shared_nameModel_2/conv2d_3/bias_1

+Model_2/conv2d_3/bias_1/Read/ReadVariableOpReadVariableOpModel_2/conv2d_3/bias_1*
_output_shapes
:*
dtype0
�
%Variable_4/Initializer/ReadVariableOpReadVariableOpModel_2/conv2d_3/bias_1*
_class
loc:@Variable_4*
_output_shapes
:*
dtype0
�

Variable_4VarHandleOp*
_class
loc:@Variable_4*
_output_shapes
: *

debug_nameVariable_4/*
dtype0*
shape:*
shared_name
Variable_4
e
+Variable_4/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_4*
_output_shapes
: 
e
Variable_4/AssignAssignVariableOp
Variable_4%Variable_4/Initializer/ReadVariableOp*
dtype0
e
Variable_4/Read/ReadVariableOpReadVariableOp
Variable_4*
_output_shapes
:*
dtype0
�
Model_2/conv2d_3/kernel_1VarHandleOp*
_output_shapes
: **

debug_nameModel_2/conv2d_3/kernel_1/*
dtype0*
shape:**
shared_nameModel_2/conv2d_3/kernel_1
�
-Model_2/conv2d_3/kernel_1/Read/ReadVariableOpReadVariableOpModel_2/conv2d_3/kernel_1*&
_output_shapes
:*
dtype0
�
%Variable_5/Initializer/ReadVariableOpReadVariableOpModel_2/conv2d_3/kernel_1*
_class
loc:@Variable_5*&
_output_shapes
:*
dtype0
�

Variable_5VarHandleOp*
_class
loc:@Variable_5*
_output_shapes
: *

debug_nameVariable_5/*
dtype0*
shape:*
shared_name
Variable_5
e
+Variable_5/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_5*
_output_shapes
: 
e
Variable_5/AssignAssignVariableOp
Variable_5%Variable_5/Initializer/ReadVariableOp*
dtype0
q
Variable_5/Read/ReadVariableOpReadVariableOp
Variable_5*&
_output_shapes
:*
dtype0
�
Model_2/conv2d_2/bias_1VarHandleOp*
_output_shapes
: *(

debug_nameModel_2/conv2d_2/bias_1/*
dtype0*
shape:*(
shared_nameModel_2/conv2d_2/bias_1

+Model_2/conv2d_2/bias_1/Read/ReadVariableOpReadVariableOpModel_2/conv2d_2/bias_1*
_output_shapes
:*
dtype0
�
%Variable_6/Initializer/ReadVariableOpReadVariableOpModel_2/conv2d_2/bias_1*
_class
loc:@Variable_6*
_output_shapes
:*
dtype0
�

Variable_6VarHandleOp*
_class
loc:@Variable_6*
_output_shapes
: *

debug_nameVariable_6/*
dtype0*
shape:*
shared_name
Variable_6
e
+Variable_6/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_6*
_output_shapes
: 
e
Variable_6/AssignAssignVariableOp
Variable_6%Variable_6/Initializer/ReadVariableOp*
dtype0
e
Variable_6/Read/ReadVariableOpReadVariableOp
Variable_6*
_output_shapes
:*
dtype0
�
Model_2/conv2d_2/kernel_1VarHandleOp*
_output_shapes
: **

debug_nameModel_2/conv2d_2/kernel_1/*
dtype0*
shape:**
shared_nameModel_2/conv2d_2/kernel_1
�
-Model_2/conv2d_2/kernel_1/Read/ReadVariableOpReadVariableOpModel_2/conv2d_2/kernel_1*&
_output_shapes
:*
dtype0
�
%Variable_7/Initializer/ReadVariableOpReadVariableOpModel_2/conv2d_2/kernel_1*
_class
loc:@Variable_7*&
_output_shapes
:*
dtype0
�

Variable_7VarHandleOp*
_class
loc:@Variable_7*
_output_shapes
: *

debug_nameVariable_7/*
dtype0*
shape:*
shared_name
Variable_7
e
+Variable_7/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_7*
_output_shapes
: 
e
Variable_7/AssignAssignVariableOp
Variable_7%Variable_7/Initializer/ReadVariableOp*
dtype0
q
Variable_7/Read/ReadVariableOpReadVariableOp
Variable_7*&
_output_shapes
:*
dtype0
�
'seed_generator_5/seed_generator_state_1VarHandleOp*
_output_shapes
: *8

debug_name*(seed_generator_5/seed_generator_state_1/*
dtype0	*
shape:*8
shared_name)'seed_generator_5/seed_generator_state_1
�
;seed_generator_5/seed_generator_state_1/Read/ReadVariableOpReadVariableOp'seed_generator_5/seed_generator_state_1*
_output_shapes
:*
dtype0	
�
%Variable_8/Initializer/ReadVariableOpReadVariableOp'seed_generator_5/seed_generator_state_1*
_class
loc:@Variable_8*
_output_shapes
:*
dtype0	
�

Variable_8VarHandleOp*
_class
loc:@Variable_8*
_output_shapes
: *

debug_nameVariable_8/*
dtype0	*
shape:*
shared_name
Variable_8
e
+Variable_8/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_8*
_output_shapes
: 
e
Variable_8/AssignAssignVariableOp
Variable_8%Variable_8/Initializer/ReadVariableOp*
dtype0	
e
Variable_8/Read/ReadVariableOpReadVariableOp
Variable_8*
_output_shapes
:*
dtype0	
�
'seed_generator_4/seed_generator_state_1VarHandleOp*
_output_shapes
: *8

debug_name*(seed_generator_4/seed_generator_state_1/*
dtype0	*
shape:*8
shared_name)'seed_generator_4/seed_generator_state_1
�
;seed_generator_4/seed_generator_state_1/Read/ReadVariableOpReadVariableOp'seed_generator_4/seed_generator_state_1*
_output_shapes
:*
dtype0	
�
%Variable_9/Initializer/ReadVariableOpReadVariableOp'seed_generator_4/seed_generator_state_1*
_class
loc:@Variable_9*
_output_shapes
:*
dtype0	
�

Variable_9VarHandleOp*
_class
loc:@Variable_9*
_output_shapes
: *

debug_nameVariable_9/*
dtype0	*
shape:*
shared_name
Variable_9
e
+Variable_9/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_9*
_output_shapes
: 
e
Variable_9/AssignAssignVariableOp
Variable_9%Variable_9/Initializer/ReadVariableOp*
dtype0	
e
Variable_9/Read/ReadVariableOpReadVariableOp
Variable_9*
_output_shapes
:*
dtype0	
�
serve_input_layer_3Placeholder*1
_output_shapes
:�����������*
dtype0*&
shape:�����������
�
StatefulPartitionedCallStatefulPartitionedCallserve_input_layer_3'seed_generator_4/seed_generator_state_1'seed_generator_5/seed_generator_state_1Model_2/conv2d_2/kernel_1Model_2/conv2d_2/bias_1Model_2/conv2d_3/kernel_1Model_2/conv2d_3/bias_1Model_2/conv2d_4/kernel_1Model_2/conv2d_4/bias_1Model_2/dense_2/kernel_1Model_2/dense_2/bias_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	
*5
config_proto%#

CPU

GPU2*0J 8� �J *3
f.R,
*__inference_signature_wrapper___call___784
�
serving_default_input_layer_3Placeholder*1
_output_shapes
:�����������*
dtype0*&
shape:�����������
�
StatefulPartitionedCall_1StatefulPartitionedCallserving_default_input_layer_3'seed_generator_4/seed_generator_state_1'seed_generator_5/seed_generator_state_1Model_2/conv2d_2/kernel_1Model_2/conv2d_2/bias_1Model_2/conv2d_3/kernel_1Model_2/conv2d_3/bias_1Model_2/conv2d_4/kernel_1Model_2/conv2d_4/bias_1Model_2/dense_2/kernel_1Model_2/dense_2/bias_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	
*5
config_proto%#

CPU

GPU2*0J 8� �J *3
f.R,
*__inference_signature_wrapper___call___809

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
	variables
trainable_variables
non_trainable_variables
_all_variables
_misc_assets
	serve

signatures*
J
0
	1

2
3
4
5
6
7
8
9*
<

0
1
2
3
4
5
6
7*

0
	1*
J
0
1
2
3
4
5
6
7
8
9*
* 

trace_0* 
"
	serve
serving_default* 
JD
VARIABLE_VALUE
Variable_9&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
Variable_8&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
Variable_7&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
Variable_6&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
Variable_5&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
Variable_4&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
Variable_3&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
Variable_2&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
Variable_1&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
HB
VARIABLE_VALUEVariable&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEModel_2/conv2d_2/kernel_1+_all_variables/0/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEModel_2/conv2d_3/kernel_1+_all_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEModel_2/conv2d_3/bias_1+_all_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEModel_2/conv2d_4/bias_1+_all_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEModel_2/dense_2/bias_1+_all_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEModel_2/conv2d_2/bias_1+_all_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEModel_2/conv2d_4/kernel_1+_all_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEModel_2/dense_2/kernel_1+_all_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE'seed_generator_4/seed_generator_state_1+_all_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE'seed_generator_5/seed_generator_state_1+_all_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename
Variable_9
Variable_8
Variable_7
Variable_6
Variable_5
Variable_4
Variable_3
Variable_2
Variable_1VariableModel_2/conv2d_2/kernel_1Model_2/conv2d_3/kernel_1Model_2/conv2d_3/bias_1Model_2/conv2d_4/bias_1Model_2/dense_2/bias_1Model_2/conv2d_2/bias_1Model_2/conv2d_4/kernel_1Model_2/dense_2/kernel_1'seed_generator_4/seed_generator_state_1'seed_generator_5/seed_generator_state_1Const*!
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *5
config_proto%#

CPU

GPU2*0J 8� �J *%
f R
__inference__traced_save_993
�
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filename
Variable_9
Variable_8
Variable_7
Variable_6
Variable_5
Variable_4
Variable_3
Variable_2
Variable_1VariableModel_2/conv2d_2/kernel_1Model_2/conv2d_3/kernel_1Model_2/conv2d_3/bias_1Model_2/conv2d_4/bias_1Model_2/dense_2/bias_1Model_2/conv2d_2/bias_1Model_2/conv2d_4/kernel_1Model_2/dense_2/kernel_1'seed_generator_4/seed_generator_state_1'seed_generator_5/seed_generator_state_1* 
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *5
config_proto%#

CPU

GPU2*0J 8� �J *)
f$R"
 __inference__traced_restore_1062��
�
�
__inference__traced_save_993
file_prefix/
!read_disablecopyonread_variable_9:	1
#read_1_disablecopyonread_variable_8:	=
#read_2_disablecopyonread_variable_7:1
#read_3_disablecopyonread_variable_6:=
#read_4_disablecopyonread_variable_5:1
#read_5_disablecopyonread_variable_4:=
#read_6_disablecopyonread_variable_3: 1
#read_7_disablecopyonread_variable_2: 7
#read_8_disablecopyonread_variable_1:
��/
!read_9_disablecopyonread_variable:M
3read_10_disablecopyonread_model_2_conv2d_2_kernel_1:M
3read_11_disablecopyonread_model_2_conv2d_3_kernel_1:?
1read_12_disablecopyonread_model_2_conv2d_3_bias_1:?
1read_13_disablecopyonread_model_2_conv2d_4_bias_1: >
0read_14_disablecopyonread_model_2_dense_2_bias_1:?
1read_15_disablecopyonread_model_2_conv2d_2_bias_1:M
3read_16_disablecopyonread_model_2_conv2d_4_kernel_1: F
2read_17_disablecopyonread_model_2_dense_2_kernel_1:
��O
Aread_18_disablecopyonread_seed_generator_4_seed_generator_state_1:	O
Aread_19_disablecopyonread_seed_generator_5_seed_generator_state_1:	
savev2_const
identity_41��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: d
Read/DisableCopyOnReadDisableCopyOnRead!read_disablecopyonread_variable_9*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp!read_disablecopyonread_variable_9^Read/DisableCopyOnRead*
_output_shapes
:*
dtype0	V
IdentityIdentityRead/ReadVariableOp:value:0*
T0	*
_output_shapes
:]

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0	*
_output_shapes
:h
Read_1/DisableCopyOnReadDisableCopyOnRead#read_1_disablecopyonread_variable_8*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp#read_1_disablecopyonread_variable_8^Read_1/DisableCopyOnRead*
_output_shapes
:*
dtype0	Z

Identity_2IdentityRead_1/ReadVariableOp:value:0*
T0	*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0	*
_output_shapes
:h
Read_2/DisableCopyOnReadDisableCopyOnRead#read_2_disablecopyonread_variable_7*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp#read_2_disablecopyonread_variable_7^Read_2/DisableCopyOnRead*&
_output_shapes
:*
dtype0f

Identity_4IdentityRead_2/ReadVariableOp:value:0*
T0*&
_output_shapes
:k

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*&
_output_shapes
:h
Read_3/DisableCopyOnReadDisableCopyOnRead#read_3_disablecopyonread_variable_6*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp#read_3_disablecopyonread_variable_6^Read_3/DisableCopyOnRead*
_output_shapes
:*
dtype0Z

Identity_6IdentityRead_3/ReadVariableOp:value:0*
T0*
_output_shapes
:_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:h
Read_4/DisableCopyOnReadDisableCopyOnRead#read_4_disablecopyonread_variable_5*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp#read_4_disablecopyonread_variable_5^Read_4/DisableCopyOnRead*&
_output_shapes
:*
dtype0f

Identity_8IdentityRead_4/ReadVariableOp:value:0*
T0*&
_output_shapes
:k

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*&
_output_shapes
:h
Read_5/DisableCopyOnReadDisableCopyOnRead#read_5_disablecopyonread_variable_4*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp#read_5_disablecopyonread_variable_4^Read_5/DisableCopyOnRead*
_output_shapes
:*
dtype0[
Identity_10IdentityRead_5/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:h
Read_6/DisableCopyOnReadDisableCopyOnRead#read_6_disablecopyonread_variable_3*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp#read_6_disablecopyonread_variable_3^Read_6/DisableCopyOnRead*&
_output_shapes
: *
dtype0g
Identity_12IdentityRead_6/ReadVariableOp:value:0*
T0*&
_output_shapes
: m
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*&
_output_shapes
: h
Read_7/DisableCopyOnReadDisableCopyOnRead#read_7_disablecopyonread_variable_2*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp#read_7_disablecopyonread_variable_2^Read_7/DisableCopyOnRead*
_output_shapes
: *
dtype0[
Identity_14IdentityRead_7/ReadVariableOp:value:0*
T0*
_output_shapes
: a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
: h
Read_8/DisableCopyOnReadDisableCopyOnRead#read_8_disablecopyonread_variable_1*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp#read_8_disablecopyonread_variable_1^Read_8/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0a
Identity_16IdentityRead_8/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��g
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��f
Read_9/DisableCopyOnReadDisableCopyOnRead!read_9_disablecopyonread_variable*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp!read_9_disablecopyonread_variable^Read_9/DisableCopyOnRead*
_output_shapes
:*
dtype0[
Identity_18IdentityRead_9/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:y
Read_10/DisableCopyOnReadDisableCopyOnRead3read_10_disablecopyonread_model_2_conv2d_2_kernel_1*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp3read_10_disablecopyonread_model_2_conv2d_2_kernel_1^Read_10/DisableCopyOnRead*&
_output_shapes
:*
dtype0h
Identity_20IdentityRead_10/ReadVariableOp:value:0*
T0*&
_output_shapes
:m
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*&
_output_shapes
:y
Read_11/DisableCopyOnReadDisableCopyOnRead3read_11_disablecopyonread_model_2_conv2d_3_kernel_1*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp3read_11_disablecopyonread_model_2_conv2d_3_kernel_1^Read_11/DisableCopyOnRead*&
_output_shapes
:*
dtype0h
Identity_22IdentityRead_11/ReadVariableOp:value:0*
T0*&
_output_shapes
:m
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*&
_output_shapes
:w
Read_12/DisableCopyOnReadDisableCopyOnRead1read_12_disablecopyonread_model_2_conv2d_3_bias_1*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp1read_12_disablecopyonread_model_2_conv2d_3_bias_1^Read_12/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_24IdentityRead_12/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:w
Read_13/DisableCopyOnReadDisableCopyOnRead1read_13_disablecopyonread_model_2_conv2d_4_bias_1*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp1read_13_disablecopyonread_model_2_conv2d_4_bias_1^Read_13/DisableCopyOnRead*
_output_shapes
: *
dtype0\
Identity_26IdentityRead_13/ReadVariableOp:value:0*
T0*
_output_shapes
: a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_14/DisableCopyOnReadDisableCopyOnRead0read_14_disablecopyonread_model_2_dense_2_bias_1*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp0read_14_disablecopyonread_model_2_dense_2_bias_1^Read_14/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_28IdentityRead_14/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:w
Read_15/DisableCopyOnReadDisableCopyOnRead1read_15_disablecopyonread_model_2_conv2d_2_bias_1*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp1read_15_disablecopyonread_model_2_conv2d_2_bias_1^Read_15/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_30IdentityRead_15/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:y
Read_16/DisableCopyOnReadDisableCopyOnRead3read_16_disablecopyonread_model_2_conv2d_4_kernel_1*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp3read_16_disablecopyonread_model_2_conv2d_4_kernel_1^Read_16/DisableCopyOnRead*&
_output_shapes
: *
dtype0h
Identity_32IdentityRead_16/ReadVariableOp:value:0*
T0*&
_output_shapes
: m
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*&
_output_shapes
: x
Read_17/DisableCopyOnReadDisableCopyOnRead2read_17_disablecopyonread_model_2_dense_2_kernel_1*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp2read_17_disablecopyonread_model_2_dense_2_kernel_1^Read_17/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0b
Identity_34IdentityRead_17/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��g
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_18/DisableCopyOnReadDisableCopyOnReadAread_18_disablecopyonread_seed_generator_4_seed_generator_state_1*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOpAread_18_disablecopyonread_seed_generator_4_seed_generator_state_1^Read_18/DisableCopyOnRead*
_output_shapes
:*
dtype0	\
Identity_36IdentityRead_18/ReadVariableOp:value:0*
T0	*
_output_shapes
:a
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0	*
_output_shapes
:�
Read_19/DisableCopyOnReadDisableCopyOnReadAread_19_disablecopyonread_seed_generator_5_seed_generator_state_1*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOpAread_19_disablecopyonread_seed_generator_5_seed_generator_state_1^Read_19/DisableCopyOnRead*
_output_shapes
:*
dtype0	\
Identity_38IdentityRead_19/ReadVariableOp:value:0*
T0	*
_output_shapes
:a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0	*
_output_shapes
:L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/0/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/1/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/2/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/3/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/4/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/5/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/6/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/7/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/8/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/9/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *#
dtypes
2				�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_40Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_41IdentityIdentity_40:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_41Identity_41:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,: : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=9

_output_shapes
: 

_user_specified_nameConst:GC
A
_user_specified_name)'seed_generator_5/seed_generator_state_1:GC
A
_user_specified_name)'seed_generator_4/seed_generator_state_1:84
2
_user_specified_nameModel_2/dense_2/kernel_1:95
3
_user_specified_nameModel_2/conv2d_4/kernel_1:73
1
_user_specified_nameModel_2/conv2d_2/bias_1:62
0
_user_specified_nameModel_2/dense_2/bias_1:73
1
_user_specified_nameModel_2/conv2d_4/bias_1:73
1
_user_specified_nameModel_2/conv2d_3/bias_1:95
3
_user_specified_nameModel_2/conv2d_3/kernel_1:95
3
_user_specified_nameModel_2/conv2d_2/kernel_1:(
$
"
_user_specified_name
Variable:*	&
$
_user_specified_name
Variable_1:*&
$
_user_specified_name
Variable_2:*&
$
_user_specified_name
Variable_3:*&
$
_user_specified_name
Variable_4:*&
$
_user_specified_name
Variable_5:*&
$
_user_specified_name
Variable_6:*&
$
_user_specified_name
Variable_7:*&
$
_user_specified_name
Variable_8:*&
$
_user_specified_name
Variable_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
��
�
__inference___call___758
input_layer_3J
<model_2_1_sequential_1_random_flip_1_readvariableop_resource:	N
@model_2_1_sequential_1_random_rotation_1_readvariableop_resource:	R
8model_2_1_conv2d_2_1_convolution_readvariableop_resource:B
4model_2_1_conv2d_2_1_reshape_readvariableop_resource:R
8model_2_1_conv2d_3_1_convolution_readvariableop_resource:B
4model_2_1_conv2d_3_1_reshape_readvariableop_resource:R
8model_2_1_conv2d_4_1_convolution_readvariableop_resource: B
4model_2_1_conv2d_4_1_reshape_readvariableop_resource: D
0model_2_1_dense_2_1_cast_readvariableop_resource:
��A
3model_2_1_dense_2_1_biasadd_readvariableop_resource:
identity��+Model_2_1/conv2d_2_1/Reshape/ReadVariableOp�/Model_2_1/conv2d_2_1/convolution/ReadVariableOp�+Model_2_1/conv2d_3_1/Reshape/ReadVariableOp�/Model_2_1/conv2d_3_1/convolution/ReadVariableOp�+Model_2_1/conv2d_4_1/Reshape/ReadVariableOp�/Model_2_1/conv2d_4_1/convolution/ReadVariableOp�*Model_2_1/dense_2_1/BiasAdd/ReadVariableOp�'Model_2_1/dense_2_1/Cast/ReadVariableOp�7Model_2_1/sequential_1/random_flip_1/Add/ReadVariableOp�5Model_2_1/sequential_1/random_flip_1/AssignVariableOp�3Model_2_1/sequential_1/random_flip_1/ReadVariableOp�;Model_2_1/sequential_1/random_rotation_1/Add/ReadVariableOp�9Model_2_1/sequential_1/random_rotation_1/AssignVariableOp�7Model_2_1/sequential_1/random_rotation_1/ReadVariableOpc
Model_2_1/rescaling_2_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *���;e
 Model_2_1/rescaling_2_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    `
Model_2_1/rescaling_2_1/ShapeConst*
_output_shapes
: *
dtype0*
valueB �
Model_2_1/rescaling_2_1/mulMulinput_layer_3'Model_2_1/rescaling_2_1/Cast/x:output:0*
T0*1
_output_shapes
:������������
Model_2_1/rescaling_2_1/addAddV2Model_2_1/rescaling_2_1/mul:z:0)Model_2_1/rescaling_2_1/Cast_1/x:output:0*
T0*1
_output_shapes
:������������
*Model_2_1/sequential_1/random_flip_1/ShapeShapeModel_2_1/rescaling_2_1/add:z:0*
T0*
_output_shapes
::���
8Model_2_1/sequential_1/random_flip_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
:Model_2_1/sequential_1/random_flip_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
:Model_2_1/sequential_1/random_flip_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
2Model_2_1/sequential_1/random_flip_1/strided_sliceStridedSlice3Model_2_1/sequential_1/random_flip_1/Shape:output:0AModel_2_1/sequential_1/random_flip_1/strided_slice/stack:output:0CModel_2_1/sequential_1/random_flip_1/strided_slice/stack_1:output:0CModel_2_1/sequential_1/random_flip_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
3Model_2_1/sequential_1/random_flip_1/ReadVariableOpReadVariableOp<model_2_1_sequential_1_random_flip_1_readvariableop_resource*
_output_shapes
:*
dtype0	l
*Model_2_1/sequential_1/random_flip_1/mul/yConst*
_output_shapes
: *
dtype0	*
value	B	 R�
(Model_2_1/sequential_1/random_flip_1/mulMul;Model_2_1/sequential_1/random_flip_1/ReadVariableOp:value:03Model_2_1/sequential_1/random_flip_1/mul/y:output:0*
T0	*
_output_shapes
:�
*Model_2_1/sequential_1/random_flip_1/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"               �
7Model_2_1/sequential_1/random_flip_1/Add/ReadVariableOpReadVariableOp<model_2_1_sequential_1_random_flip_1_readvariableop_resource*
_output_shapes
:*
dtype0	�
(Model_2_1/sequential_1/random_flip_1/AddAddV2?Model_2_1/sequential_1/random_flip_1/Add/ReadVariableOp:value:03Model_2_1/sequential_1/random_flip_1/Const:output:0*
T0	*
_output_shapes
:�
5Model_2_1/sequential_1/random_flip_1/AssignVariableOpAssignVariableOp<model_2_1_sequential_1_random_flip_1_readvariableop_resource,Model_2_1/sequential_1/random_flip_1/Add:z:08^Model_2_1/sequential_1/random_flip_1/Add/ReadVariableOp4^Model_2_1/sequential_1/random_flip_1/ReadVariableOp*
_output_shapes
 *
dtype0	*
validate_shape(u
/Model_2_1/sequential_1/random_flip_1/FloorMod/yConst*
_output_shapes
: *
dtype0	*
valueB	 R�����
-Model_2_1/sequential_1/random_flip_1/FloorModFloorMod,Model_2_1/sequential_1/random_flip_1/mul:z:08Model_2_1/sequential_1/random_flip_1/FloorMod/y:output:0*
T0	*
_output_shapes
:�
)Model_2_1/sequential_1/random_flip_1/CastCast1Model_2_1/sequential_1/random_flip_1/FloorMod:z:0*

DstT0*

SrcT0	*
_output_shapes
:r
-Model_2_1/sequential_1/random_flip_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    r
-Model_2_1/sequential_1/random_flip_1/Cast_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
EModel_2_1/sequential_1/random_flip_1/stateless_random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :�
EModel_2_1/sequential_1/random_flip_1/stateless_random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
EModel_2_1/sequential_1/random_flip_1/stateless_random_uniform/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
CModel_2_1/sequential_1/random_flip_1/stateless_random_uniform/shapePack;Model_2_1/sequential_1/random_flip_1/strided_slice:output:0NModel_2_1/sequential_1/random_flip_1/stateless_random_uniform/shape/1:output:0NModel_2_1/sequential_1/random_flip_1/stateless_random_uniform/shape/2:output:0NModel_2_1/sequential_1/random_flip_1/stateless_random_uniform/shape/3:output:0*
N*
T0*
_output_shapes
:�
ZModel_2_1/sequential_1/random_flip_1/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter-Model_2_1/sequential_1/random_flip_1/Cast:y:0*
Tseed0* 
_output_shapes
::�
ZModel_2_1/sequential_1/random_flip_1/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :�
VModel_2_1/sequential_1/random_flip_1/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2LModel_2_1/sequential_1/random_flip_1/stateless_random_uniform/shape:output:0`Model_2_1/sequential_1/random_flip_1/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0dModel_2_1/sequential_1/random_flip_1/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0cModel_2_1/sequential_1/random_flip_1/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*/
_output_shapes
:����������
AModel_2_1/sequential_1/random_flip_1/stateless_random_uniform/subSub6Model_2_1/sequential_1/random_flip_1/Cast_2/x:output:06Model_2_1/sequential_1/random_flip_1/Cast_1/x:output:0*
T0*
_output_shapes
: �
AModel_2_1/sequential_1/random_flip_1/stateless_random_uniform/mulMul_Model_2_1/sequential_1/random_flip_1/stateless_random_uniform/StatelessRandomUniformV2:output:0EModel_2_1/sequential_1/random_flip_1/stateless_random_uniform/sub:z:0*
T0*/
_output_shapes
:����������
=Model_2_1/sequential_1/random_flip_1/stateless_random_uniformAddV2EModel_2_1/sequential_1/random_flip_1/stateless_random_uniform/mul:z:06Model_2_1/sequential_1/random_flip_1/Cast_1/x:output:0*
T0*/
_output_shapes
:���������q
,Model_2_1/sequential_1/random_flip_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?�
.Model_2_1/sequential_1/random_flip_1/LessEqual	LessEqualAModel_2_1/sequential_1/random_flip_1/stateless_random_uniform:z:05Model_2_1/sequential_1/random_flip_1/Const_1:output:0*
T0*/
_output_shapes
:����������
,Model_2_1/sequential_1/random_flip_1/Shape_1ShapeModel_2_1/rescaling_2_1/add:z:0*
T0*
_output_shapes
::���
:Model_2_1/sequential_1/random_flip_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
<Model_2_1/sequential_1/random_flip_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
<Model_2_1/sequential_1/random_flip_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
4Model_2_1/sequential_1/random_flip_1/strided_slice_1StridedSlice5Model_2_1/sequential_1/random_flip_1/Shape_1:output:0CModel_2_1/sequential_1/random_flip_1/strided_slice_1/stack:output:0EModel_2_1/sequential_1/random_flip_1/strided_slice_1/stack_1:output:0EModel_2_1/sequential_1/random_flip_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
,Model_2_1/sequential_1/random_flip_1/Shape_2ShapeModel_2_1/rescaling_2_1/add:z:0*
T0*
_output_shapes
::���
:Model_2_1/sequential_1/random_flip_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: �
<Model_2_1/sequential_1/random_flip_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
<Model_2_1/sequential_1/random_flip_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
4Model_2_1/sequential_1/random_flip_1/strided_slice_2StridedSlice5Model_2_1/sequential_1/random_flip_1/Shape_2:output:0CModel_2_1/sequential_1/random_flip_1/strided_slice_2/stack:output:0EModel_2_1/sequential_1/random_flip_1/strided_slice_2/stack_1:output:0EModel_2_1/sequential_1/random_flip_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
3Model_2_1/sequential_1/random_flip_1/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:
����������
.Model_2_1/sequential_1/random_flip_1/ReverseV2	ReverseV2Model_2_1/rescaling_2_1/add:z:0<Model_2_1/sequential_1/random_flip_1/ReverseV2/axis:output:0*
T0*1
_output_shapes
:������������
-Model_2_1/sequential_1/random_flip_1/SelectV2SelectV22Model_2_1/sequential_1/random_flip_1/LessEqual:z:07Model_2_1/sequential_1/random_flip_1/ReverseV2:output:0Model_2_1/rescaling_2_1/add:z:0*
T0*1
_output_shapes
:������������
5Model_2_1/sequential_1/random_flip_1/ReverseV2_1/axisConst*
_output_shapes
:*
dtype0*
valueB:
����������
0Model_2_1/sequential_1/random_flip_1/ReverseV2_1	ReverseV26Model_2_1/sequential_1/random_flip_1/SelectV2:output:0>Model_2_1/sequential_1/random_flip_1/ReverseV2_1/axis:output:0*
T0*1
_output_shapes
:������������
/Model_2_1/sequential_1/random_flip_1/SelectV2_1SelectV22Model_2_1/sequential_1/random_flip_1/LessEqual:z:09Model_2_1/sequential_1/random_flip_1/ReverseV2_1:output:06Model_2_1/sequential_1/random_flip_1/SelectV2:output:0*
T0*1
_output_shapes
:������������
.Model_2_1/sequential_1/random_rotation_1/ShapeShape8Model_2_1/sequential_1/random_flip_1/SelectV2_1:output:0*
T0*
_output_shapes
::���
<Model_2_1/sequential_1/random_rotation_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
>Model_2_1/sequential_1/random_rotation_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
>Model_2_1/sequential_1/random_rotation_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
6Model_2_1/sequential_1/random_rotation_1/strided_sliceStridedSlice7Model_2_1/sequential_1/random_rotation_1/Shape:output:0EModel_2_1/sequential_1/random_rotation_1/strided_slice/stack:output:0GModel_2_1/sequential_1/random_rotation_1/strided_slice/stack_1:output:0GModel_2_1/sequential_1/random_rotation_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
7Model_2_1/sequential_1/random_rotation_1/ReadVariableOpReadVariableOp@model_2_1_sequential_1_random_rotation_1_readvariableop_resource*
_output_shapes
:*
dtype0	p
.Model_2_1/sequential_1/random_rotation_1/mul/yConst*
_output_shapes
: *
dtype0	*
value	B	 R�
,Model_2_1/sequential_1/random_rotation_1/mulMul?Model_2_1/sequential_1/random_rotation_1/ReadVariableOp:value:07Model_2_1/sequential_1/random_rotation_1/mul/y:output:0*
T0	*
_output_shapes
:�
.Model_2_1/sequential_1/random_rotation_1/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"               �
;Model_2_1/sequential_1/random_rotation_1/Add/ReadVariableOpReadVariableOp@model_2_1_sequential_1_random_rotation_1_readvariableop_resource*
_output_shapes
:*
dtype0	�
,Model_2_1/sequential_1/random_rotation_1/AddAddV2CModel_2_1/sequential_1/random_rotation_1/Add/ReadVariableOp:value:07Model_2_1/sequential_1/random_rotation_1/Const:output:0*
T0	*
_output_shapes
:�
9Model_2_1/sequential_1/random_rotation_1/AssignVariableOpAssignVariableOp@model_2_1_sequential_1_random_rotation_1_readvariableop_resource0Model_2_1/sequential_1/random_rotation_1/Add:z:0<^Model_2_1/sequential_1/random_rotation_1/Add/ReadVariableOp8^Model_2_1/sequential_1/random_rotation_1/ReadVariableOp*
_output_shapes
 *
dtype0	*
validate_shape(y
3Model_2_1/sequential_1/random_rotation_1/FloorMod/yConst*
_output_shapes
: *
dtype0	*
valueB	 R�����
1Model_2_1/sequential_1/random_rotation_1/FloorModFloorMod0Model_2_1/sequential_1/random_rotation_1/mul:z:0<Model_2_1/sequential_1/random_rotation_1/FloorMod/y:output:0*
T0	*
_output_shapes
:�
-Model_2_1/sequential_1/random_rotation_1/CastCast5Model_2_1/sequential_1/random_rotation_1/FloorMod:z:0*

DstT0*

SrcT0	*
_output_shapes
:v
1Model_2_1/sequential_1/random_rotation_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �v
1Model_2_1/sequential_1/random_rotation_1/Cast_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  C�
GModel_2_1/sequential_1/random_rotation_1/stateless_random_uniform/shapePack?Model_2_1/sequential_1/random_rotation_1/strided_slice:output:0*
N*
T0*
_output_shapes
:�
^Model_2_1/sequential_1/random_rotation_1/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter1Model_2_1/sequential_1/random_rotation_1/Cast:y:0*
Tseed0* 
_output_shapes
::�
^Model_2_1/sequential_1/random_rotation_1/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :�
ZModel_2_1/sequential_1/random_rotation_1/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2PModel_2_1/sequential_1/random_rotation_1/stateless_random_uniform/shape:output:0dModel_2_1/sequential_1/random_rotation_1/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0hModel_2_1/sequential_1/random_rotation_1/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0gModel_2_1/sequential_1/random_rotation_1/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:����������
EModel_2_1/sequential_1/random_rotation_1/stateless_random_uniform/subSub:Model_2_1/sequential_1/random_rotation_1/Cast_2/x:output:0:Model_2_1/sequential_1/random_rotation_1/Cast_1/x:output:0*
T0*
_output_shapes
: �
EModel_2_1/sequential_1/random_rotation_1/stateless_random_uniform/mulMulcModel_2_1/sequential_1/random_rotation_1/stateless_random_uniform/StatelessRandomUniformV2:output:0IModel_2_1/sequential_1/random_rotation_1/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:����������
AModel_2_1/sequential_1/random_rotation_1/stateless_random_uniformAddV2IModel_2_1/sequential_1/random_rotation_1/stateless_random_uniform/mul:z:0:Model_2_1/sequential_1/random_rotation_1/Cast_1/x:output:0*
T0*#
_output_shapes
:����������
5Model_2_1/sequential_1/random_rotation_1/zeros/packedPack?Model_2_1/sequential_1/random_rotation_1/strided_slice:output:0*
N*
T0*
_output_shapes
:y
4Model_2_1/sequential_1/random_rotation_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.Model_2_1/sequential_1/random_rotation_1/zerosFill>Model_2_1/sequential_1/random_rotation_1/zeros/packed:output:0=Model_2_1/sequential_1/random_rotation_1/zeros/Const:output:0*
T0*#
_output_shapes
:����������
7Model_2_1/sequential_1/random_rotation_1/zeros_1/packedPack?Model_2_1/sequential_1/random_rotation_1/strided_slice:output:0*
N*
T0*
_output_shapes
:{
6Model_2_1/sequential_1/random_rotation_1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
0Model_2_1/sequential_1/random_rotation_1/zeros_1Fill@Model_2_1/sequential_1/random_rotation_1/zeros_1/packed:output:0?Model_2_1/sequential_1/random_rotation_1/zeros_1/Const:output:0*
T0*#
_output_shapes
:����������
4Model_2_1/sequential_1/random_rotation_1/ones/packedPack?Model_2_1/sequential_1/random_rotation_1/strided_slice:output:0*
N*
T0*
_output_shapes
:x
3Model_2_1/sequential_1/random_rotation_1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
-Model_2_1/sequential_1/random_rotation_1/onesFill=Model_2_1/sequential_1/random_rotation_1/ones/packed:output:0<Model_2_1/sequential_1/random_rotation_1/ones/Const:output:0*
T0*#
_output_shapes
:����������
7Model_2_1/sequential_1/random_rotation_1/zeros_2/packedPack?Model_2_1/sequential_1/random_rotation_1/strided_slice:output:0*
N*
T0*
_output_shapes
:{
6Model_2_1/sequential_1/random_rotation_1/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
0Model_2_1/sequential_1/random_rotation_1/zeros_2Fill@Model_2_1/sequential_1/random_rotation_1/zeros_2/packed:output:0?Model_2_1/sequential_1/random_rotation_1/zeros_2/Const:output:0*
T0*#
_output_shapes
:����������
7Model_2_1/sequential_1/random_rotation_1/zeros_3/packedPack?Model_2_1/sequential_1/random_rotation_1/strided_slice:output:0*
N*
T0*
_output_shapes
:{
6Model_2_1/sequential_1/random_rotation_1/zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
0Model_2_1/sequential_1/random_rotation_1/zeros_3Fill@Model_2_1/sequential_1/random_rotation_1/zeros_3/packed:output:0?Model_2_1/sequential_1/random_rotation_1/zeros_3/Const:output:0*
T0*#
_output_shapes
:���������u
0Model_2_1/sequential_1/random_rotation_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *5��<�
.Model_2_1/sequential_1/random_rotation_1/mul_1MulEModel_2_1/sequential_1/random_rotation_1/stateless_random_uniform:z:09Model_2_1/sequential_1/random_rotation_1/Const_1:output:0*
T0*#
_output_shapes
:����������
.Model_2_1/sequential_1/random_rotation_1/mul_2Mul9Model_2_1/sequential_1/random_rotation_1/zeros_2:output:09Model_2_1/sequential_1/random_rotation_1/Const_1:output:0*
T0*#
_output_shapes
:����������
.Model_2_1/sequential_1/random_rotation_1/mul_3Mul9Model_2_1/sequential_1/random_rotation_1/zeros_3:output:09Model_2_1/sequential_1/random_rotation_1/Const_1:output:0*
T0*#
_output_shapes
:����������
0Model_2_1/sequential_1/random_rotation_1/Shape_1Shape2Model_2_1/sequential_1/random_rotation_1/mul_1:z:0*
T0*
_output_shapes
::���
>Model_2_1/sequential_1/random_rotation_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
@Model_2_1/sequential_1/random_rotation_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
@Model_2_1/sequential_1/random_rotation_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
8Model_2_1/sequential_1/random_rotation_1/strided_slice_1StridedSlice9Model_2_1/sequential_1/random_rotation_1/Shape_1:output:0GModel_2_1/sequential_1/random_rotation_1/strided_slice_1/stack:output:0IModel_2_1/sequential_1/random_rotation_1/strided_slice_1/stack_1:output:0IModel_2_1/sequential_1/random_rotation_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
1Model_2_1/sequential_1/random_rotation_1/Cast_3/xConst*
_output_shapes
: *
dtype0*
value
B :��
/Model_2_1/sequential_1/random_rotation_1/Cast_3Cast:Model_2_1/sequential_1/random_rotation_1/Cast_3/x:output:0*

DstT0*

SrcT0*
_output_shapes
: t
1Model_2_1/sequential_1/random_rotation_1/Cast_4/xConst*
_output_shapes
: *
dtype0*
value
B :��
/Model_2_1/sequential_1/random_rotation_1/Cast_4Cast:Model_2_1/sequential_1/random_rotation_1/Cast_4/x:output:0*

DstT0*

SrcT0*
_output_shapes
: s
.Model_2_1/sequential_1/random_rotation_1/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
,Model_2_1/sequential_1/random_rotation_1/subSub3Model_2_1/sequential_1/random_rotation_1/Cast_3:y:07Model_2_1/sequential_1/random_rotation_1/sub/y:output:0*
T0*
_output_shapes
: u
0Model_2_1/sequential_1/random_rotation_1/mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
.Model_2_1/sequential_1/random_rotation_1/mul_4Mul9Model_2_1/sequential_1/random_rotation_1/mul_4/x:output:00Model_2_1/sequential_1/random_rotation_1/sub:z:0*
T0*
_output_shapes
: u
0Model_2_1/sequential_1/random_rotation_1/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
.Model_2_1/sequential_1/random_rotation_1/sub_1Sub3Model_2_1/sequential_1/random_rotation_1/Cast_4:y:09Model_2_1/sequential_1/random_rotation_1/sub_1/y:output:0*
T0*
_output_shapes
: u
0Model_2_1/sequential_1/random_rotation_1/mul_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
.Model_2_1/sequential_1/random_rotation_1/mul_5Mul9Model_2_1/sequential_1/random_rotation_1/mul_5/x:output:02Model_2_1/sequential_1/random_rotation_1/sub_1:z:0*
T0*
_output_shapes
: �
,Model_2_1/sequential_1/random_rotation_1/CosCos2Model_2_1/sequential_1/random_rotation_1/mul_1:z:0*
T0*#
_output_shapes
:����������
,Model_2_1/sequential_1/random_rotation_1/SinSin2Model_2_1/sequential_1/random_rotation_1/mul_1:z:0*
T0*#
_output_shapes
:����������
,Model_2_1/sequential_1/random_rotation_1/TanTan2Model_2_1/sequential_1/random_rotation_1/mul_2:z:0*
T0*#
_output_shapes
:����������
.Model_2_1/sequential_1/random_rotation_1/Tan_1Tan2Model_2_1/sequential_1/random_rotation_1/mul_3:z:0*
T0*#
_output_shapes
:����������
.Model_2_1/sequential_1/random_rotation_1/mul_6Mul0Model_2_1/sequential_1/random_rotation_1/Tan:y:00Model_2_1/sequential_1/random_rotation_1/Sin:y:0*
T0*#
_output_shapes
:����������
.Model_2_1/sequential_1/random_rotation_1/add_1AddV20Model_2_1/sequential_1/random_rotation_1/Cos:y:02Model_2_1/sequential_1/random_rotation_1/mul_6:z:0*
T0*#
_output_shapes
:����������
.Model_2_1/sequential_1/random_rotation_1/mul_7Mul6Model_2_1/sequential_1/random_rotation_1/ones:output:02Model_2_1/sequential_1/random_rotation_1/add_1:z:0*
T0*#
_output_shapes
:����������
,Model_2_1/sequential_1/random_rotation_1/NegNeg0Model_2_1/sequential_1/random_rotation_1/Sin:y:0*
T0*#
_output_shapes
:����������
.Model_2_1/sequential_1/random_rotation_1/mul_8Mul0Model_2_1/sequential_1/random_rotation_1/Tan:y:00Model_2_1/sequential_1/random_rotation_1/Cos:y:0*
T0*#
_output_shapes
:����������
.Model_2_1/sequential_1/random_rotation_1/add_2AddV20Model_2_1/sequential_1/random_rotation_1/Neg:y:02Model_2_1/sequential_1/random_rotation_1/mul_8:z:0*
T0*#
_output_shapes
:����������
.Model_2_1/sequential_1/random_rotation_1/mul_9Mul6Model_2_1/sequential_1/random_rotation_1/ones:output:02Model_2_1/sequential_1/random_rotation_1/add_2:z:0*
T0*#
_output_shapes
:����������
.Model_2_1/sequential_1/random_rotation_1/add_3AddV27Model_2_1/sequential_1/random_rotation_1/zeros:output:02Model_2_1/sequential_1/random_rotation_1/mul_4:z:0*
T0*#
_output_shapes
:����������
/Model_2_1/sequential_1/random_rotation_1/mul_10Mul2Model_2_1/sequential_1/random_rotation_1/mul_4:z:02Model_2_1/sequential_1/random_rotation_1/mul_7:z:0*
T0*#
_output_shapes
:����������
.Model_2_1/sequential_1/random_rotation_1/sub_2Sub2Model_2_1/sequential_1/random_rotation_1/add_3:z:03Model_2_1/sequential_1/random_rotation_1/mul_10:z:0*
T0*#
_output_shapes
:����������
/Model_2_1/sequential_1/random_rotation_1/mul_11Mul2Model_2_1/sequential_1/random_rotation_1/mul_5:z:02Model_2_1/sequential_1/random_rotation_1/mul_9:z:0*
T0*#
_output_shapes
:����������
.Model_2_1/sequential_1/random_rotation_1/sub_3Sub2Model_2_1/sequential_1/random_rotation_1/sub_2:z:03Model_2_1/sequential_1/random_rotation_1/mul_11:z:0*
T0*#
_output_shapes
:����������
/Model_2_1/sequential_1/random_rotation_1/mul_12Mul2Model_2_1/sequential_1/random_rotation_1/Tan_1:y:00Model_2_1/sequential_1/random_rotation_1/Cos:y:0*
T0*#
_output_shapes
:����������
.Model_2_1/sequential_1/random_rotation_1/add_4AddV23Model_2_1/sequential_1/random_rotation_1/mul_12:z:00Model_2_1/sequential_1/random_rotation_1/Sin:y:0*
T0*#
_output_shapes
:����������
/Model_2_1/sequential_1/random_rotation_1/mul_13Mul6Model_2_1/sequential_1/random_rotation_1/ones:output:02Model_2_1/sequential_1/random_rotation_1/add_4:z:0*
T0*#
_output_shapes
:����������
.Model_2_1/sequential_1/random_rotation_1/Neg_1Neg0Model_2_1/sequential_1/random_rotation_1/Sin:y:0*
T0*#
_output_shapes
:����������
/Model_2_1/sequential_1/random_rotation_1/mul_14Mul2Model_2_1/sequential_1/random_rotation_1/Tan_1:y:02Model_2_1/sequential_1/random_rotation_1/Neg_1:y:0*
T0*#
_output_shapes
:����������
.Model_2_1/sequential_1/random_rotation_1/add_5AddV23Model_2_1/sequential_1/random_rotation_1/mul_14:z:00Model_2_1/sequential_1/random_rotation_1/Cos:y:0*
T0*#
_output_shapes
:����������
/Model_2_1/sequential_1/random_rotation_1/mul_15Mul6Model_2_1/sequential_1/random_rotation_1/ones:output:02Model_2_1/sequential_1/random_rotation_1/add_5:z:0*
T0*#
_output_shapes
:����������
.Model_2_1/sequential_1/random_rotation_1/add_6AddV29Model_2_1/sequential_1/random_rotation_1/zeros_1:output:02Model_2_1/sequential_1/random_rotation_1/mul_5:z:0*
T0*#
_output_shapes
:����������
/Model_2_1/sequential_1/random_rotation_1/mul_16Mul2Model_2_1/sequential_1/random_rotation_1/mul_4:z:03Model_2_1/sequential_1/random_rotation_1/mul_13:z:0*
T0*#
_output_shapes
:����������
.Model_2_1/sequential_1/random_rotation_1/sub_4Sub2Model_2_1/sequential_1/random_rotation_1/add_6:z:03Model_2_1/sequential_1/random_rotation_1/mul_16:z:0*
T0*#
_output_shapes
:����������
/Model_2_1/sequential_1/random_rotation_1/mul_17Mul2Model_2_1/sequential_1/random_rotation_1/mul_5:z:03Model_2_1/sequential_1/random_rotation_1/mul_15:z:0*
T0*#
_output_shapes
:����������
.Model_2_1/sequential_1/random_rotation_1/sub_5Sub2Model_2_1/sequential_1/random_rotation_1/sub_4:z:03Model_2_1/sequential_1/random_rotation_1/mul_17:z:0*
T0*#
_output_shapes
:����������
>Model_2_1/sequential_1/random_rotation_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
@Model_2_1/sequential_1/random_rotation_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        �
@Model_2_1/sequential_1/random_rotation_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
8Model_2_1/sequential_1/random_rotation_1/strided_slice_2StridedSlice2Model_2_1/sequential_1/random_rotation_1/mul_7:z:0GModel_2_1/sequential_1/random_rotation_1/strided_slice_2/stack:output:0IModel_2_1/sequential_1/random_rotation_1/strided_slice_2/stack_1:output:0IModel_2_1/sequential_1/random_rotation_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask�
>Model_2_1/sequential_1/random_rotation_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
@Model_2_1/sequential_1/random_rotation_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        �
@Model_2_1/sequential_1/random_rotation_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
8Model_2_1/sequential_1/random_rotation_1/strided_slice_3StridedSlice2Model_2_1/sequential_1/random_rotation_1/mul_9:z:0GModel_2_1/sequential_1/random_rotation_1/strided_slice_3/stack:output:0IModel_2_1/sequential_1/random_rotation_1/strided_slice_3/stack_1:output:0IModel_2_1/sequential_1/random_rotation_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask�
>Model_2_1/sequential_1/random_rotation_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
@Model_2_1/sequential_1/random_rotation_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        �
@Model_2_1/sequential_1/random_rotation_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
8Model_2_1/sequential_1/random_rotation_1/strided_slice_4StridedSlice2Model_2_1/sequential_1/random_rotation_1/sub_3:z:0GModel_2_1/sequential_1/random_rotation_1/strided_slice_4/stack:output:0IModel_2_1/sequential_1/random_rotation_1/strided_slice_4/stack_1:output:0IModel_2_1/sequential_1/random_rotation_1/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask�
>Model_2_1/sequential_1/random_rotation_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
@Model_2_1/sequential_1/random_rotation_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        �
@Model_2_1/sequential_1/random_rotation_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
8Model_2_1/sequential_1/random_rotation_1/strided_slice_5StridedSlice3Model_2_1/sequential_1/random_rotation_1/mul_13:z:0GModel_2_1/sequential_1/random_rotation_1/strided_slice_5/stack:output:0IModel_2_1/sequential_1/random_rotation_1/strided_slice_5/stack_1:output:0IModel_2_1/sequential_1/random_rotation_1/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask�
>Model_2_1/sequential_1/random_rotation_1/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
@Model_2_1/sequential_1/random_rotation_1/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        �
@Model_2_1/sequential_1/random_rotation_1/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
8Model_2_1/sequential_1/random_rotation_1/strided_slice_6StridedSlice3Model_2_1/sequential_1/random_rotation_1/mul_15:z:0GModel_2_1/sequential_1/random_rotation_1/strided_slice_6/stack:output:0IModel_2_1/sequential_1/random_rotation_1/strided_slice_6/stack_1:output:0IModel_2_1/sequential_1/random_rotation_1/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask�
>Model_2_1/sequential_1/random_rotation_1/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
@Model_2_1/sequential_1/random_rotation_1/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        �
@Model_2_1/sequential_1/random_rotation_1/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
8Model_2_1/sequential_1/random_rotation_1/strided_slice_7StridedSlice2Model_2_1/sequential_1/random_rotation_1/sub_5:z:0GModel_2_1/sequential_1/random_rotation_1/strided_slice_7/stack:output:0IModel_2_1/sequential_1/random_rotation_1/strided_slice_7/stack_1:output:0IModel_2_1/sequential_1/random_rotation_1/strided_slice_7/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask{
9Model_2_1/sequential_1/random_rotation_1/zeros_4/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
7Model_2_1/sequential_1/random_rotation_1/zeros_4/packedPackAModel_2_1/sequential_1/random_rotation_1/strided_slice_1:output:0BModel_2_1/sequential_1/random_rotation_1/zeros_4/packed/1:output:0*
N*
T0*
_output_shapes
:{
6Model_2_1/sequential_1/random_rotation_1/zeros_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
0Model_2_1/sequential_1/random_rotation_1/zeros_4Fill@Model_2_1/sequential_1/random_rotation_1/zeros_4/packed:output:0?Model_2_1/sequential_1/random_rotation_1/zeros_4/Const:output:0*
T0*'
_output_shapes
:���������v
4Model_2_1/sequential_1/random_rotation_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
/Model_2_1/sequential_1/random_rotation_1/concatConcatV2AModel_2_1/sequential_1/random_rotation_1/strided_slice_2:output:0AModel_2_1/sequential_1/random_rotation_1/strided_slice_3:output:0AModel_2_1/sequential_1/random_rotation_1/strided_slice_4:output:0AModel_2_1/sequential_1/random_rotation_1/strided_slice_5:output:0AModel_2_1/sequential_1/random_rotation_1/strided_slice_6:output:0AModel_2_1/sequential_1/random_rotation_1/strided_slice_7:output:09Model_2_1/sequential_1/random_rotation_1/zeros_4:output:0=Model_2_1/sequential_1/random_rotation_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:����������
0Model_2_1/sequential_1/random_rotation_1/Shape_2Shape8Model_2_1/sequential_1/random_flip_1/SelectV2_1:output:0*
T0*
_output_shapes
::���
>Model_2_1/sequential_1/random_rotation_1/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: �
@Model_2_1/sequential_1/random_rotation_1/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
@Model_2_1/sequential_1/random_rotation_1/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
8Model_2_1/sequential_1/random_rotation_1/strided_slice_8StridedSlice9Model_2_1/sequential_1/random_rotation_1/Shape_2:output:0GModel_2_1/sequential_1/random_rotation_1/strided_slice_8/stack:output:0IModel_2_1/sequential_1/random_rotation_1/strided_slice_8/stack_1:output:0IModel_2_1/sequential_1/random_rotation_1/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
0Model_2_1/sequential_1/random_rotation_1/Shape_3Shape8Model_2_1/sequential_1/random_flip_1/SelectV2_1:output:0*
T0*
_output_shapes
::���
>Model_2_1/sequential_1/random_rotation_1/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:�
@Model_2_1/sequential_1/random_rotation_1/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
����������
@Model_2_1/sequential_1/random_rotation_1/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
8Model_2_1/sequential_1/random_rotation_1/strided_slice_9StridedSlice9Model_2_1/sequential_1/random_rotation_1/Shape_3:output:0GModel_2_1/sequential_1/random_rotation_1/strided_slice_9/stack:output:0IModel_2_1/sequential_1/random_rotation_1/strided_slice_9/stack_1:output:0IModel_2_1/sequential_1/random_rotation_1/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
:�
NModel_2_1/sequential_1/random_rotation_1/ImageProjectiveTransformV3/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
CModel_2_1/sequential_1/random_rotation_1/ImageProjectiveTransformV3ImageProjectiveTransformV38Model_2_1/sequential_1/random_flip_1/SelectV2_1:output:08Model_2_1/sequential_1/random_rotation_1/concat:output:0AModel_2_1/sequential_1/random_rotation_1/strided_slice_9:output:0WModel_2_1/sequential_1/random_rotation_1/ImageProjectiveTransformV3/fill_value:output:0*A
_output_shapes/
-:+���������������������������*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR�
4Model_2_1/sequential_1/random_rotation_1/EnsureShapeEnsureShapeXModel_2_1/sequential_1/random_rotation_1/ImageProjectiveTransformV3:transformed_images:0*
T0*1
_output_shapes
:�����������*&
shape:������������
/Model_2_1/conv2d_2_1/convolution/ReadVariableOpReadVariableOp8model_2_1_conv2d_2_1_convolution_readvariableop_resource*&
_output_shapes
:*
dtype0�
 Model_2_1/conv2d_2_1/convolutionConv2D=Model_2_1/sequential_1/random_rotation_1/EnsureShape:output:07Model_2_1/conv2d_2_1/convolution/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
+Model_2_1/conv2d_2_1/Reshape/ReadVariableOpReadVariableOp4model_2_1_conv2d_2_1_reshape_readvariableop_resource*
_output_shapes
:*
dtype0{
"Model_2_1/conv2d_2_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            �
Model_2_1/conv2d_2_1/ReshapeReshape3Model_2_1/conv2d_2_1/Reshape/ReadVariableOp:value:0+Model_2_1/conv2d_2_1/Reshape/shape:output:0*
T0*&
_output_shapes
:s
Model_2_1/conv2d_2_1/SqueezeSqueeze%Model_2_1/conv2d_2_1/Reshape:output:0*
T0*
_output_shapes
:�
Model_2_1/conv2d_2_1/BiasAddBiasAdd)Model_2_1/conv2d_2_1/convolution:output:0%Model_2_1/conv2d_2_1/Squeeze:output:0*
T0*1
_output_shapes
:������������
Model_2_1/conv2d_2_1/ReluRelu%Model_2_1/conv2d_2_1/BiasAdd:output:0*
T0*1
_output_shapes
:������������
%Model_2_1/max_pooling2d_2_1/MaxPool2dMaxPool'Model_2_1/conv2d_2_1/Relu:activations:0*/
_output_shapes
:���������pp*
ksize
*
paddingVALID*
strides
�
/Model_2_1/conv2d_3_1/convolution/ReadVariableOpReadVariableOp8model_2_1_conv2d_3_1_convolution_readvariableop_resource*&
_output_shapes
:*
dtype0�
 Model_2_1/conv2d_3_1/convolutionConv2D.Model_2_1/max_pooling2d_2_1/MaxPool2d:output:07Model_2_1/conv2d_3_1/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp*
paddingSAME*
strides
�
+Model_2_1/conv2d_3_1/Reshape/ReadVariableOpReadVariableOp4model_2_1_conv2d_3_1_reshape_readvariableop_resource*
_output_shapes
:*
dtype0{
"Model_2_1/conv2d_3_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            �
Model_2_1/conv2d_3_1/ReshapeReshape3Model_2_1/conv2d_3_1/Reshape/ReadVariableOp:value:0+Model_2_1/conv2d_3_1/Reshape/shape:output:0*
T0*&
_output_shapes
:s
Model_2_1/conv2d_3_1/SqueezeSqueeze%Model_2_1/conv2d_3_1/Reshape:output:0*
T0*
_output_shapes
:�
Model_2_1/conv2d_3_1/BiasAddBiasAdd)Model_2_1/conv2d_3_1/convolution:output:0%Model_2_1/conv2d_3_1/Squeeze:output:0*
T0*/
_output_shapes
:���������pp�
Model_2_1/conv2d_3_1/ReluRelu%Model_2_1/conv2d_3_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������pp�
%Model_2_1/max_pooling2d_3_1/MaxPool2dMaxPool'Model_2_1/conv2d_3_1/Relu:activations:0*/
_output_shapes
:���������88*
ksize
*
paddingVALID*
strides
�
/Model_2_1/conv2d_4_1/convolution/ReadVariableOpReadVariableOp8model_2_1_conv2d_4_1_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0�
 Model_2_1/conv2d_4_1/convolutionConv2D.Model_2_1/max_pooling2d_3_1/MaxPool2d:output:07Model_2_1/conv2d_4_1/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88 *
paddingSAME*
strides
�
+Model_2_1/conv2d_4_1/Reshape/ReadVariableOpReadVariableOp4model_2_1_conv2d_4_1_reshape_readvariableop_resource*
_output_shapes
: *
dtype0{
"Model_2_1/conv2d_4_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             �
Model_2_1/conv2d_4_1/ReshapeReshape3Model_2_1/conv2d_4_1/Reshape/ReadVariableOp:value:0+Model_2_1/conv2d_4_1/Reshape/shape:output:0*
T0*&
_output_shapes
: s
Model_2_1/conv2d_4_1/SqueezeSqueeze%Model_2_1/conv2d_4_1/Reshape:output:0*
T0*
_output_shapes
: �
Model_2_1/conv2d_4_1/BiasAddBiasAdd)Model_2_1/conv2d_4_1/convolution:output:0%Model_2_1/conv2d_4_1/Squeeze:output:0*
T0*/
_output_shapes
:���������88 �
Model_2_1/conv2d_4_1/ReluRelu%Model_2_1/conv2d_4_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������88 �
%Model_2_1/max_pooling2d_4_1/MaxPool2dMaxPool'Model_2_1/conv2d_4_1/Relu:activations:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
t
#Model_2_1/flatten_2_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"���� b  �
Model_2_1/flatten_2_1/ReshapeReshape.Model_2_1/max_pooling2d_4_1/MaxPool2d:output:0,Model_2_1/flatten_2_1/Reshape/shape:output:0*
T0*)
_output_shapes
:������������
'Model_2_1/dense_2_1/Cast/ReadVariableOpReadVariableOp0model_2_1_dense_2_1_cast_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
Model_2_1/dense_2_1/MatMulMatMul&Model_2_1/flatten_2_1/Reshape:output:0/Model_2_1/dense_2_1/Cast/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
*Model_2_1/dense_2_1/BiasAdd/ReadVariableOpReadVariableOp3model_2_1_dense_2_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Model_2_1/dense_2_1/BiasAddBiasAdd$Model_2_1/dense_2_1/MatMul:product:02Model_2_1/dense_2_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������~
Model_2_1/dense_2_1/SoftmaxSoftmax$Model_2_1/dense_2_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������t
IdentityIdentity%Model_2_1/dense_2_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp,^Model_2_1/conv2d_2_1/Reshape/ReadVariableOp0^Model_2_1/conv2d_2_1/convolution/ReadVariableOp,^Model_2_1/conv2d_3_1/Reshape/ReadVariableOp0^Model_2_1/conv2d_3_1/convolution/ReadVariableOp,^Model_2_1/conv2d_4_1/Reshape/ReadVariableOp0^Model_2_1/conv2d_4_1/convolution/ReadVariableOp+^Model_2_1/dense_2_1/BiasAdd/ReadVariableOp(^Model_2_1/dense_2_1/Cast/ReadVariableOp8^Model_2_1/sequential_1/random_flip_1/Add/ReadVariableOp6^Model_2_1/sequential_1/random_flip_1/AssignVariableOp4^Model_2_1/sequential_1/random_flip_1/ReadVariableOp<^Model_2_1/sequential_1/random_rotation_1/Add/ReadVariableOp:^Model_2_1/sequential_1/random_rotation_1/AssignVariableOp8^Model_2_1/sequential_1/random_rotation_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:�����������: : : : : : : : : : 2Z
+Model_2_1/conv2d_2_1/Reshape/ReadVariableOp+Model_2_1/conv2d_2_1/Reshape/ReadVariableOp2b
/Model_2_1/conv2d_2_1/convolution/ReadVariableOp/Model_2_1/conv2d_2_1/convolution/ReadVariableOp2Z
+Model_2_1/conv2d_3_1/Reshape/ReadVariableOp+Model_2_1/conv2d_3_1/Reshape/ReadVariableOp2b
/Model_2_1/conv2d_3_1/convolution/ReadVariableOp/Model_2_1/conv2d_3_1/convolution/ReadVariableOp2Z
+Model_2_1/conv2d_4_1/Reshape/ReadVariableOp+Model_2_1/conv2d_4_1/Reshape/ReadVariableOp2b
/Model_2_1/conv2d_4_1/convolution/ReadVariableOp/Model_2_1/conv2d_4_1/convolution/ReadVariableOp2X
*Model_2_1/dense_2_1/BiasAdd/ReadVariableOp*Model_2_1/dense_2_1/BiasAdd/ReadVariableOp2R
'Model_2_1/dense_2_1/Cast/ReadVariableOp'Model_2_1/dense_2_1/Cast/ReadVariableOp2r
7Model_2_1/sequential_1/random_flip_1/Add/ReadVariableOp7Model_2_1/sequential_1/random_flip_1/Add/ReadVariableOp2n
5Model_2_1/sequential_1/random_flip_1/AssignVariableOp5Model_2_1/sequential_1/random_flip_1/AssignVariableOp2j
3Model_2_1/sequential_1/random_flip_1/ReadVariableOp3Model_2_1/sequential_1/random_flip_1/ReadVariableOp2z
;Model_2_1/sequential_1/random_rotation_1/Add/ReadVariableOp;Model_2_1/sequential_1/random_rotation_1/Add/ReadVariableOp2v
9Model_2_1/sequential_1/random_rotation_1/AssignVariableOp9Model_2_1/sequential_1/random_rotation_1/AssignVariableOp2r
7Model_2_1/sequential_1/random_rotation_1/ReadVariableOp7Model_2_1/sequential_1/random_rotation_1/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:` \
1
_output_shapes
:�����������
'
_user_specified_nameinput_layer_3
�^
�
 __inference__traced_restore_1062
file_prefix)
assignvariableop_variable_9:	+
assignvariableop_1_variable_8:	7
assignvariableop_2_variable_7:+
assignvariableop_3_variable_6:7
assignvariableop_4_variable_5:+
assignvariableop_5_variable_4:7
assignvariableop_6_variable_3: +
assignvariableop_7_variable_2: 1
assignvariableop_8_variable_1:
��)
assignvariableop_9_variable:G
-assignvariableop_10_model_2_conv2d_2_kernel_1:G
-assignvariableop_11_model_2_conv2d_3_kernel_1:9
+assignvariableop_12_model_2_conv2d_3_bias_1:9
+assignvariableop_13_model_2_conv2d_4_bias_1: 8
*assignvariableop_14_model_2_dense_2_bias_1:9
+assignvariableop_15_model_2_conv2d_2_bias_1:G
-assignvariableop_16_model_2_conv2d_4_kernel_1: @
,assignvariableop_17_model_2_dense_2_kernel_1:
��I
;assignvariableop_18_seed_generator_4_seed_generator_state_1:	I
;assignvariableop_19_seed_generator_5_seed_generator_state_1:	
identity_21��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/0/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/1/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/2/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/3/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/4/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/5/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/6/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/7/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/8/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/9/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*h
_output_shapesV
T:::::::::::::::::::::*#
dtypes
2				[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_variable_9Identity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_variable_8Identity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_variable_7Identity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_variable_6Identity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_variable_5Identity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_variable_4Identity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_variable_3Identity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_variable_2Identity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_variable_1Identity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_variableIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp-assignvariableop_10_model_2_conv2d_2_kernel_1Identity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp-assignvariableop_11_model_2_conv2d_3_kernel_1Identity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp+assignvariableop_12_model_2_conv2d_3_bias_1Identity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp+assignvariableop_13_model_2_conv2d_4_bias_1Identity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp*assignvariableop_14_model_2_dense_2_bias_1Identity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp+assignvariableop_15_model_2_conv2d_2_bias_1Identity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp-assignvariableop_16_model_2_conv2d_4_kernel_1Identity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp,assignvariableop_17_model_2_dense_2_kernel_1Identity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp;assignvariableop_18_seed_generator_4_seed_generator_state_1Identity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp;assignvariableop_19_seed_generator_5_seed_generator_state_1Identity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_20Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_21IdentityIdentity_20:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_21Identity_21:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*: : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:GC
A
_user_specified_name)'seed_generator_5/seed_generator_state_1:GC
A
_user_specified_name)'seed_generator_4/seed_generator_state_1:84
2
_user_specified_nameModel_2/dense_2/kernel_1:95
3
_user_specified_nameModel_2/conv2d_4/kernel_1:73
1
_user_specified_nameModel_2/conv2d_2/bias_1:62
0
_user_specified_nameModel_2/dense_2/bias_1:73
1
_user_specified_nameModel_2/conv2d_4/bias_1:73
1
_user_specified_nameModel_2/conv2d_3/bias_1:95
3
_user_specified_nameModel_2/conv2d_3/kernel_1:95
3
_user_specified_nameModel_2/conv2d_2/kernel_1:(
$
"
_user_specified_name
Variable:*	&
$
_user_specified_name
Variable_1:*&
$
_user_specified_name
Variable_2:*&
$
_user_specified_name
Variable_3:*&
$
_user_specified_name
Variable_4:*&
$
_user_specified_name
Variable_5:*&
$
_user_specified_name
Variable_6:*&
$
_user_specified_name
Variable_7:*&
$
_user_specified_name
Variable_8:*&
$
_user_specified_name
Variable_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
*__inference_signature_wrapper___call___809
input_layer_3
unknown:	
	unknown_0:	#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5: 
	unknown_6: 
	unknown_7:
��
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_layer_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	
*5
config_proto%#

CPU

GPU2*0J 8� �J *!
fR
__inference___call___758o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:�����������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:#


_user_specified_name805:#	

_user_specified_name803:#

_user_specified_name801:#

_user_specified_name799:#

_user_specified_name797:#

_user_specified_name795:#

_user_specified_name793:#

_user_specified_name791:#

_user_specified_name789:#

_user_specified_name787:` \
1
_output_shapes
:�����������
'
_user_specified_nameinput_layer_3
�
�
*__inference_signature_wrapper___call___784
input_layer_3
unknown:	
	unknown_0:	#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5: 
	unknown_6: 
	unknown_7:
��
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_layer_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	
*5
config_proto%#

CPU

GPU2*0J 8� �J *!
fR
__inference___call___758o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:�����������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:#


_user_specified_name780:#	

_user_specified_name778:#

_user_specified_name776:#

_user_specified_name774:#

_user_specified_name772:#

_user_specified_name770:#

_user_specified_name768:#

_user_specified_name766:#

_user_specified_name764:#

_user_specified_name762:` \
1
_output_shapes
:�����������
'
_user_specified_nameinput_layer_3"�L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serve�
G
input_layer_36
serve_input_layer_3:0�����������<
output_00
StatefulPartitionedCall:0���������tensorflow/serving/predict*�
serving_default�
Q
input_layer_3@
serving_default_input_layer_3:0�����������>
output_02
StatefulPartitionedCall_1:0���������tensorflow/serving/predict:�
�
	variables
trainable_variables
non_trainable_variables
_all_variables
_misc_assets
	serve

signatures"
_generic_user_object
f
0
	1

2
3
4
5
6
7
8
9"
trackable_list_wrapper
X

0
1
2
3
4
5
6
7"
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
8
9"
trackable_list_wrapper
 "
trackable_list_wrapper
�
trace_02�
__inference___call___758�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *6�3
1�.
input_layer_3�����������ztrace_0
7
	serve
serving_default"
signature_map
1:/	2%seed_generator_4/seed_generator_state
1:/	2%seed_generator_5/seed_generator_state
1:/2Model_2/conv2d_2/kernel
#:!2Model_2/conv2d_2/bias
1:/2Model_2/conv2d_3/kernel
#:!2Model_2/conv2d_3/bias
1:/ 2Model_2/conv2d_4/kernel
#:! 2Model_2/conv2d_4/bias
*:(
��2Model_2/dense_2/kernel
": 2Model_2/dense_2/bias
1:/2Model_2/conv2d_2/kernel
1:/2Model_2/conv2d_3/kernel
#:!2Model_2/conv2d_3/bias
#:! 2Model_2/conv2d_4/bias
": 2Model_2/dense_2/bias
#:!2Model_2/conv2d_2/bias
1:/ 2Model_2/conv2d_4/kernel
*:(
��2Model_2/dense_2/kernel
1:/	2%seed_generator_4/seed_generator_state
1:/	2%seed_generator_5/seed_generator_state
�B�
__inference___call___758input_layer_3"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
*__inference_signature_wrapper___call___784input_layer_3"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 "

kwonlyargs�
jinput_layer_3
kwonlydefaults
 
annotations� *
 
�B�
*__inference_signature_wrapper___call___809input_layer_3"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 "

kwonlyargs�
jinput_layer_3
kwonlydefaults
 
annotations� *
 �
__inference___call___758q
	
@�=
6�3
1�.
input_layer_3�����������
� "!�
unknown����������
*__inference_signature_wrapper___call___784�
	
Q�N
� 
G�D
B
input_layer_31�.
input_layer_3�����������"3�0
.
output_0"�
output_0����������
*__inference_signature_wrapper___call___809�
	
Q�N
� 
G�D
B
input_layer_31�.
input_layer_3�����������"3�0
.
output_0"�
output_0���������