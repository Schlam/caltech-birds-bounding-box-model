0
!á 
B
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
¼
AvgPool

value"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype:
2
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

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

DepthwiseConv2dNative

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

ú
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%·Ñ8"&
exponential_avg_factorfloat%  ?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
9
	IdentityN

input2T
output2T"
T
list(type)(0
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

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
:
Maximum
x"T
y"T
z"T"
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
_
Pad

input"T
paddings"	Tpaddings
output"T"	
Ttype"
	Tpaddingstype0:
2	
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
-
Sqrt
x"T
y"T"
Ttype:

2
¾
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
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.12v2.4.0-49-g85c8b2a817f8Ê&
|
normalization/meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namenormalization/mean
u
&normalization/mean/Read/ReadVariableOpReadVariableOpnormalization/mean*
_output_shapes
:*
dtype0

normalization/varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namenormalization/variance
}
*normalization/variance/Read/ReadVariableOpReadVariableOpnormalization/variance*
_output_shapes
:*
dtype0
z
normalization/countVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *$
shared_namenormalization/count
s
'normalization/count/Read/ReadVariableOpReadVariableOpnormalization/count*
_output_shapes
: *
dtype0	

stem_conv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*!
shared_namestem_conv/kernel
}
$stem_conv/kernel/Read/ReadVariableOpReadVariableOpstem_conv/kernel*&
_output_shapes
:(*
dtype0
r
stem_bn/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_namestem_bn/gamma
k
!stem_bn/gamma/Read/ReadVariableOpReadVariableOpstem_bn/gamma*
_output_shapes
:(*
dtype0
p
stem_bn/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_namestem_bn/beta
i
 stem_bn/beta/Read/ReadVariableOpReadVariableOpstem_bn/beta*
_output_shapes
:(*
dtype0
~
stem_bn/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*$
shared_namestem_bn/moving_mean
w
'stem_bn/moving_mean/Read/ReadVariableOpReadVariableOpstem_bn/moving_mean*
_output_shapes
:(*
dtype0

stem_bn/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*(
shared_namestem_bn/moving_variance

+stem_bn/moving_variance/Read/ReadVariableOpReadVariableOpstem_bn/moving_variance*
_output_shapes
:(*
dtype0
¢
block1a_dwconv/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*0
shared_name!block1a_dwconv/depthwise_kernel

3block1a_dwconv/depthwise_kernel/Read/ReadVariableOpReadVariableOpblock1a_dwconv/depthwise_kernel*&
_output_shapes
:(*
dtype0
x
block1a_bn/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*!
shared_nameblock1a_bn/gamma
q
$block1a_bn/gamma/Read/ReadVariableOpReadVariableOpblock1a_bn/gamma*
_output_shapes
:(*
dtype0
v
block1a_bn/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:(* 
shared_nameblock1a_bn/beta
o
#block1a_bn/beta/Read/ReadVariableOpReadVariableOpblock1a_bn/beta*
_output_shapes
:(*
dtype0

block1a_bn/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*'
shared_nameblock1a_bn/moving_mean
}
*block1a_bn/moving_mean/Read/ReadVariableOpReadVariableOpblock1a_bn/moving_mean*
_output_shapes
:(*
dtype0

block1a_bn/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*+
shared_nameblock1a_bn/moving_variance

.block1a_bn/moving_variance/Read/ReadVariableOpReadVariableOpblock1a_bn/moving_variance*
_output_shapes
:(*
dtype0

block1a_se_reduce/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:(
*)
shared_nameblock1a_se_reduce/kernel

,block1a_se_reduce/kernel/Read/ReadVariableOpReadVariableOpblock1a_se_reduce/kernel*&
_output_shapes
:(
*
dtype0

block1a_se_reduce/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameblock1a_se_reduce/bias
}
*block1a_se_reduce/bias/Read/ReadVariableOpReadVariableOpblock1a_se_reduce/bias*
_output_shapes
:
*
dtype0

block1a_se_expand/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
(*)
shared_nameblock1a_se_expand/kernel

,block1a_se_expand/kernel/Read/ReadVariableOpReadVariableOpblock1a_se_expand/kernel*&
_output_shapes
:
(*
dtype0

block1a_se_expand/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*'
shared_nameblock1a_se_expand/bias
}
*block1a_se_expand/bias/Read/ReadVariableOpReadVariableOpblock1a_se_expand/bias*
_output_shapes
:(*
dtype0

block1a_project_conv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*,
shared_nameblock1a_project_conv/kernel

/block1a_project_conv/kernel/Read/ReadVariableOpReadVariableOpblock1a_project_conv/kernel*&
_output_shapes
:(*
dtype0

block1a_project_bn/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameblock1a_project_bn/gamma

,block1a_project_bn/gamma/Read/ReadVariableOpReadVariableOpblock1a_project_bn/gamma*
_output_shapes
:*
dtype0

block1a_project_bn/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameblock1a_project_bn/beta

+block1a_project_bn/beta/Read/ReadVariableOpReadVariableOpblock1a_project_bn/beta*
_output_shapes
:*
dtype0

block1a_project_bn/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name block1a_project_bn/moving_mean

2block1a_project_bn/moving_mean/Read/ReadVariableOpReadVariableOpblock1a_project_bn/moving_mean*
_output_shapes
:*
dtype0

"block1a_project_bn/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"block1a_project_bn/moving_variance

6block1a_project_bn/moving_variance/Read/ReadVariableOpReadVariableOp"block1a_project_bn/moving_variance*
_output_shapes
:*
dtype0
¢
block1b_dwconv/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!block1b_dwconv/depthwise_kernel

3block1b_dwconv/depthwise_kernel/Read/ReadVariableOpReadVariableOpblock1b_dwconv/depthwise_kernel*&
_output_shapes
:*
dtype0
x
block1b_bn/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameblock1b_bn/gamma
q
$block1b_bn/gamma/Read/ReadVariableOpReadVariableOpblock1b_bn/gamma*
_output_shapes
:*
dtype0
v
block1b_bn/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameblock1b_bn/beta
o
#block1b_bn/beta/Read/ReadVariableOpReadVariableOpblock1b_bn/beta*
_output_shapes
:*
dtype0

block1b_bn/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameblock1b_bn/moving_mean
}
*block1b_bn/moving_mean/Read/ReadVariableOpReadVariableOpblock1b_bn/moving_mean*
_output_shapes
:*
dtype0

block1b_bn/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameblock1b_bn/moving_variance

.block1b_bn/moving_variance/Read/ReadVariableOpReadVariableOpblock1b_bn/moving_variance*
_output_shapes
:*
dtype0

block1b_se_reduce/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameblock1b_se_reduce/kernel

,block1b_se_reduce/kernel/Read/ReadVariableOpReadVariableOpblock1b_se_reduce/kernel*&
_output_shapes
:*
dtype0

block1b_se_reduce/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameblock1b_se_reduce/bias
}
*block1b_se_reduce/bias/Read/ReadVariableOpReadVariableOpblock1b_se_reduce/bias*
_output_shapes
:*
dtype0

block1b_se_expand/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameblock1b_se_expand/kernel

,block1b_se_expand/kernel/Read/ReadVariableOpReadVariableOpblock1b_se_expand/kernel*&
_output_shapes
:*
dtype0

block1b_se_expand/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameblock1b_se_expand/bias
}
*block1b_se_expand/bias/Read/ReadVariableOpReadVariableOpblock1b_se_expand/bias*
_output_shapes
:*
dtype0

block1b_project_conv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameblock1b_project_conv/kernel

/block1b_project_conv/kernel/Read/ReadVariableOpReadVariableOpblock1b_project_conv/kernel*&
_output_shapes
:*
dtype0

block1b_project_bn/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameblock1b_project_bn/gamma

,block1b_project_bn/gamma/Read/ReadVariableOpReadVariableOpblock1b_project_bn/gamma*
_output_shapes
:*
dtype0

block1b_project_bn/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameblock1b_project_bn/beta

+block1b_project_bn/beta/Read/ReadVariableOpReadVariableOpblock1b_project_bn/beta*
_output_shapes
:*
dtype0

block1b_project_bn/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name block1b_project_bn/moving_mean

2block1b_project_bn/moving_mean/Read/ReadVariableOpReadVariableOpblock1b_project_bn/moving_mean*
_output_shapes
:*
dtype0

"block1b_project_bn/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"block1b_project_bn/moving_variance

6block1b_project_bn/moving_variance/Read/ReadVariableOpReadVariableOp"block1b_project_bn/moving_variance*
_output_shapes
:*
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
y
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
y
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense_3/kernel
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
dtype0
x
output1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nameoutput1/kernel
q
"output1/kernel/Read/ReadVariableOpReadVariableOpoutput1/kernel*
_output_shapes

:*
dtype0
p
output1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameoutput1/bias
i
 output1/bias/Read/ReadVariableOpReadVariableOpoutput1/bias*
_output_shapes
:*
dtype0
x
output2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nameoutput2/kernel
q
"output2/kernel/Read/ReadVariableOpReadVariableOpoutput2/kernel*
_output_shapes

:*
dtype0
p
output2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameoutput2/bias
i
 output2/bias/Read/ReadVariableOpReadVariableOpoutput2/bias*
_output_shapes
:*
dtype0
x
output3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nameoutput3/kernel
q
"output3/kernel/Read/ReadVariableOpReadVariableOpoutput3/kernel*
_output_shapes

:*
dtype0
p
output3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameoutput3/bias
i
 output3/bias/Read/ReadVariableOpReadVariableOpoutput3/bias*
_output_shapes
:*
dtype0
x
output4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nameoutput4/kernel
q
"output4/kernel/Read/ReadVariableOpReadVariableOpoutput4/kernel*
_output_shapes

:*
dtype0
p
output4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameoutput4/bias
i
 output4/bias/Read/ReadVariableOpReadVariableOpoutput4/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_3
[
total_3/Read/ReadVariableOpReadVariableOptotal_3*
_output_shapes
: *
dtype0
b
count_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0
b
total_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_4
[
total_4/Read/ReadVariableOpReadVariableOptotal_4*
_output_shapes
: *
dtype0
b
count_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_4
[
count_4/Read/ReadVariableOpReadVariableOpcount_4*
_output_shapes
: *
dtype0

Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/m

(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
:*
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
:*
dtype0

Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*$
shared_nameAdam/dense/kernel/m
|
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes
:	*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:*
dtype0

Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes
:	*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:*
dtype0

Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes
:	*
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:*
dtype0

Adam/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_nameAdam/dense_3/kernel/m

)Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/m*
_output_shapes
:	*
dtype0
~
Adam/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/m
w
'Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/m*
_output_shapes
:*
dtype0

Adam/output1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/output1/kernel/m

)Adam/output1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output1/kernel/m*
_output_shapes

:*
dtype0
~
Adam/output1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/output1/bias/m
w
'Adam/output1/bias/m/Read/ReadVariableOpReadVariableOpAdam/output1/bias/m*
_output_shapes
:*
dtype0

Adam/output2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/output2/kernel/m

)Adam/output2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output2/kernel/m*
_output_shapes

:*
dtype0
~
Adam/output2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/output2/bias/m
w
'Adam/output2/bias/m/Read/ReadVariableOpReadVariableOpAdam/output2/bias/m*
_output_shapes
:*
dtype0

Adam/output3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/output3/kernel/m

)Adam/output3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output3/kernel/m*
_output_shapes

:*
dtype0
~
Adam/output3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/output3/bias/m
w
'Adam/output3/bias/m/Read/ReadVariableOpReadVariableOpAdam/output3/bias/m*
_output_shapes
:*
dtype0

Adam/output4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/output4/kernel/m

)Adam/output4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output4/kernel/m*
_output_shapes

:*
dtype0
~
Adam/output4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/output4/bias/m
w
'Adam/output4/bias/m/Read/ReadVariableOpReadVariableOpAdam/output4/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/v

(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
:*
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
:*
dtype0

Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*$
shared_nameAdam/dense/kernel/v
|
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes
:	*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:*
dtype0

Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes
:	*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:*
dtype0

Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes
:	*
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:*
dtype0

Adam/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_nameAdam/dense_3/kernel/v

)Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/v*
_output_shapes
:	*
dtype0
~
Adam/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/v
w
'Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/v*
_output_shapes
:*
dtype0

Adam/output1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/output1/kernel/v

)Adam/output1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output1/kernel/v*
_output_shapes

:*
dtype0
~
Adam/output1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/output1/bias/v
w
'Adam/output1/bias/v/Read/ReadVariableOpReadVariableOpAdam/output1/bias/v*
_output_shapes
:*
dtype0

Adam/output2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/output2/kernel/v

)Adam/output2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output2/kernel/v*
_output_shapes

:*
dtype0
~
Adam/output2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/output2/bias/v
w
'Adam/output2/bias/v/Read/ReadVariableOpReadVariableOpAdam/output2/bias/v*
_output_shapes
:*
dtype0

Adam/output3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/output3/kernel/v

)Adam/output3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output3/kernel/v*
_output_shapes

:*
dtype0
~
Adam/output3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/output3/bias/v
w
'Adam/output3/bias/v/Read/ReadVariableOpReadVariableOpAdam/output3/bias/v*
_output_shapes
:*
dtype0

Adam/output4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/output4/kernel/v

)Adam/output4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output4/kernel/v*
_output_shapes

:*
dtype0
~
Adam/output4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/output4/bias/v
w
'Adam/output4/bias/v/Read/ReadVariableOpReadVariableOpAdam/output4/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
ËÖ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ö
valueúÕBöÕ BîÕ
­

layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer-10
layer-11
layer_with_weights-5
layer-12
layer_with_weights-6
layer-13
layer-14
layer_with_weights-7
layer-15
layer_with_weights-8
layer-16
layer_with_weights-9
layer-17
layer_with_weights-10
layer-18
layer-19
layer-20
layer-21
layer_with_weights-11
layer-22
layer_with_weights-12
layer-23
layer-24
layer_with_weights-13
layer-25
layer_with_weights-14
layer-26
layer-27
layer-28
layer-29
layer_with_weights-15
layer-30
 layer-31
!layer-32
"layer_with_weights-16
"layer-33
#layer_with_weights-17
#layer-34
$layer_with_weights-18
$layer-35
%layer_with_weights-19
%layer-36
&layer_with_weights-20
&layer-37
'layer_with_weights-21
'layer-38
(layer_with_weights-22
(layer-39
)layer_with_weights-23
)layer-40
*	optimizer
+loss
,regularization_losses
-	variables
.trainable_variables
/	keras_api
0
signatures
 

1	keras_api
]
2state_variables
3_broadcast_shape
4mean
5variance
	6count
7	keras_api
R
8regularization_losses
9	variables
:trainable_variables
;	keras_api
^

<kernel
=regularization_losses
>	variables
?trainable_variables
@	keras_api

Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
R
Jregularization_losses
K	variables
Ltrainable_variables
M	keras_api
h
Ndepthwise_kernel
Oregularization_losses
P	variables
Qtrainable_variables
R	keras_api

Saxis
	Tgamma
Ubeta
Vmoving_mean
Wmoving_variance
Xregularization_losses
Y	variables
Ztrainable_variables
[	keras_api
R
\regularization_losses
]	variables
^trainable_variables
_	keras_api
R
`regularization_losses
a	variables
btrainable_variables
c	keras_api
R
dregularization_losses
e	variables
ftrainable_variables
g	keras_api
h

hkernel
ibias
jregularization_losses
k	variables
ltrainable_variables
m	keras_api
h

nkernel
obias
pregularization_losses
q	variables
rtrainable_variables
s	keras_api
R
tregularization_losses
u	variables
vtrainable_variables
w	keras_api
^

xkernel
yregularization_losses
z	variables
{trainable_variables
|	keras_api

}axis
	~gamma
beta
moving_mean
moving_variance
regularization_losses
	variables
trainable_variables
	keras_api
m
depthwise_kernel
regularization_losses
	variables
trainable_variables
	keras_api
 
	axis

gamma
	beta
moving_mean
moving_variance
regularization_losses
	variables
trainable_variables
	keras_api
V
regularization_losses
	variables
trainable_variables
	keras_api
V
regularization_losses
	variables
trainable_variables
	keras_api
V
regularization_losses
	variables
trainable_variables
	keras_api
n
 kernel
	¡bias
¢regularization_losses
£	variables
¤trainable_variables
¥	keras_api
n
¦kernel
	§bias
¨regularization_losses
©	variables
ªtrainable_variables
«	keras_api
V
¬regularization_losses
­	variables
®trainable_variables
¯	keras_api
c
°kernel
±regularization_losses
²	variables
³trainable_variables
´	keras_api
 
	µaxis

¶gamma
	·beta
¸moving_mean
¹moving_variance
ºregularization_losses
»	variables
¼trainable_variables
½	keras_api
V
¾regularization_losses
¿	variables
Àtrainable_variables
Á	keras_api
V
Âregularization_losses
Ã	variables
Ätrainable_variables
Å	keras_api
V
Æregularization_losses
Ç	variables
Ètrainable_variables
É	keras_api
n
Êkernel
	Ëbias
Ìregularization_losses
Í	variables
Îtrainable_variables
Ï	keras_api
V
Ðregularization_losses
Ñ	variables
Òtrainable_variables
Ó	keras_api
V
Ôregularization_losses
Õ	variables
Ötrainable_variables
×	keras_api
n
Økernel
	Ùbias
Úregularization_losses
Û	variables
Ütrainable_variables
Ý	keras_api
n
Þkernel
	ßbias
àregularization_losses
á	variables
âtrainable_variables
ã	keras_api
n
äkernel
	åbias
æregularization_losses
ç	variables
ètrainable_variables
é	keras_api
n
êkernel
	ëbias
ìregularization_losses
í	variables
îtrainable_variables
ï	keras_api
n
ðkernel
	ñbias
òregularization_losses
ó	variables
ôtrainable_variables
õ	keras_api
n
ökernel
	÷bias
øregularization_losses
ù	variables
útrainable_variables
û	keras_api
n
ükernel
	ýbias
þregularization_losses
ÿ	variables
trainable_variables
	keras_api
n
kernel
	bias
regularization_losses
	variables
trainable_variables
	keras_api
Ñ
	iter
beta_1
beta_2

decay
learning_rate	Êmé	Ëmê	Ømë	Ùmì	Þmí	ßmî	ämï	åmð	êmñ	ëmò	ðmó	ñmô	ömõ	÷mö	üm÷	ýmø	mù	mú	Êvû	Ëvü	Øvý	Ùvþ	Þvÿ	ßv	äv	åv	êv	ëv	ðv	ñv	öv	÷v	üv	ýv	v	v
 
 
È
40
51
62
<3
B4
C5
D6
E7
N8
T9
U10
V11
W12
h13
i14
n15
o16
x17
~18
19
20
21
22
23
24
25
26
 27
¡28
¦29
§30
°31
¶32
·33
¸34
¹35
Ê36
Ë37
Ø38
Ù39
Þ40
ß41
ä42
å43
ê44
ë45
ð46
ñ47
ö48
÷49
ü50
ý51
52
53

Ê0
Ë1
Ø2
Ù3
Þ4
ß5
ä6
å7
ê8
ë9
ð10
ñ11
ö12
÷13
ü14
ý15
16
17
²
,regularization_losses
-	variables
.trainable_variables
non_trainable_variables
 layer_regularization_losses
layer_metrics
layers
metrics
 
 
#
4mean
5variance
	6count
 
\Z
VARIABLE_VALUEnormalization/mean4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEnormalization/variance8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEnormalization/count5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
²
8regularization_losses
9	variables
non_trainable_variables
 layer_regularization_losses
:trainable_variables
layer_metrics
layers
metrics
\Z
VARIABLE_VALUEstem_conv/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

<0
 
²
=regularization_losses
>	variables
non_trainable_variables
 layer_regularization_losses
?trainable_variables
layer_metrics
layers
metrics
 
XV
VARIABLE_VALUEstem_bn/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEstem_bn/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEstem_bn/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEstem_bn/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

B0
C1
D2
E3
 
²
Fregularization_losses
G	variables
non_trainable_variables
 layer_regularization_losses
Htrainable_variables
layer_metrics
layers
 metrics
 
 
 
²
Jregularization_losses
K	variables
¡non_trainable_variables
 ¢layer_regularization_losses
Ltrainable_variables
£layer_metrics
¤layers
¥metrics
us
VARIABLE_VALUEblock1a_dwconv/depthwise_kernel@layer_with_weights-3/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
 

N0
 
²
Oregularization_losses
P	variables
¦non_trainable_variables
 §layer_regularization_losses
Qtrainable_variables
¨layer_metrics
©layers
ªmetrics
 
[Y
VARIABLE_VALUEblock1a_bn/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEblock1a_bn/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEblock1a_bn/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEblock1a_bn/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

T0
U1
V2
W3
 
²
Xregularization_losses
Y	variables
«non_trainable_variables
 ¬layer_regularization_losses
Ztrainable_variables
­layer_metrics
®layers
¯metrics
 
 
 
²
\regularization_losses
]	variables
°non_trainable_variables
 ±layer_regularization_losses
^trainable_variables
²layer_metrics
³layers
´metrics
 
 
 
²
`regularization_losses
a	variables
µnon_trainable_variables
 ¶layer_regularization_losses
btrainable_variables
·layer_metrics
¸layers
¹metrics
 
 
 
²
dregularization_losses
e	variables
ºnon_trainable_variables
 »layer_regularization_losses
ftrainable_variables
¼layer_metrics
½layers
¾metrics
db
VARIABLE_VALUEblock1a_se_reduce/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEblock1a_se_reduce/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

h0
i1
 
²
jregularization_losses
k	variables
¿non_trainable_variables
 Àlayer_regularization_losses
ltrainable_variables
Álayer_metrics
Âlayers
Ãmetrics
db
VARIABLE_VALUEblock1a_se_expand/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEblock1a_se_expand/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

n0
o1
 
²
pregularization_losses
q	variables
Änon_trainable_variables
 Ålayer_regularization_losses
rtrainable_variables
Ælayer_metrics
Çlayers
Èmetrics
 
 
 
²
tregularization_losses
u	variables
Énon_trainable_variables
 Êlayer_regularization_losses
vtrainable_variables
Ëlayer_metrics
Ìlayers
Ímetrics
ge
VARIABLE_VALUEblock1a_project_conv/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

x0
 
²
yregularization_losses
z	variables
Înon_trainable_variables
 Ïlayer_regularization_losses
{trainable_variables
Ðlayer_metrics
Ñlayers
Òmetrics
 
ca
VARIABLE_VALUEblock1a_project_bn/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEblock1a_project_bn/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEblock1a_project_bn/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE"block1a_project_bn/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

~0
1
2
3
 
µ
regularization_losses
	variables
Ónon_trainable_variables
 Ôlayer_regularization_losses
trainable_variables
Õlayer_metrics
Ölayers
×metrics
us
VARIABLE_VALUEblock1b_dwconv/depthwise_kernel@layer_with_weights-9/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
 

0
 
µ
regularization_losses
	variables
Ønon_trainable_variables
 Ùlayer_regularization_losses
trainable_variables
Úlayer_metrics
Ûlayers
Ümetrics
 
\Z
VARIABLE_VALUEblock1b_bn/gamma6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEblock1b_bn/beta5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEblock1b_bn/moving_mean<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEblock1b_bn/moving_variance@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 
0
1
2
3
 
µ
regularization_losses
	variables
Ýnon_trainable_variables
 Þlayer_regularization_losses
trainable_variables
ßlayer_metrics
àlayers
ámetrics
 
 
 
µ
regularization_losses
	variables
ânon_trainable_variables
 ãlayer_regularization_losses
trainable_variables
älayer_metrics
ålayers
æmetrics
 
 
 
µ
regularization_losses
	variables
çnon_trainable_variables
 èlayer_regularization_losses
trainable_variables
élayer_metrics
êlayers
ëmetrics
 
 
 
µ
regularization_losses
	variables
ìnon_trainable_variables
 ílayer_regularization_losses
trainable_variables
îlayer_metrics
ïlayers
ðmetrics
ec
VARIABLE_VALUEblock1b_se_reduce/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEblock1b_se_reduce/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE
 

 0
¡1
 
µ
¢regularization_losses
£	variables
ñnon_trainable_variables
 òlayer_regularization_losses
¤trainable_variables
ólayer_metrics
ôlayers
õmetrics
ec
VARIABLE_VALUEblock1b_se_expand/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEblock1b_se_expand/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE
 

¦0
§1
 
µ
¨regularization_losses
©	variables
önon_trainable_variables
 ÷layer_regularization_losses
ªtrainable_variables
ølayer_metrics
ùlayers
úmetrics
 
 
 
µ
¬regularization_losses
­	variables
ûnon_trainable_variables
 ülayer_regularization_losses
®trainable_variables
ýlayer_metrics
þlayers
ÿmetrics
hf
VARIABLE_VALUEblock1b_project_conv/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

°0
 
µ
±regularization_losses
²	variables
non_trainable_variables
 layer_regularization_losses
³trainable_variables
layer_metrics
layers
metrics
 
db
VARIABLE_VALUEblock1b_project_bn/gamma6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEblock1b_project_bn/beta5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEblock1b_project_bn/moving_mean<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE"block1b_project_bn/moving_variance@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 
¶0
·1
¸2
¹3
 
µ
ºregularization_losses
»	variables
non_trainable_variables
 layer_regularization_losses
¼trainable_variables
layer_metrics
layers
metrics
 
 
 
µ
¾regularization_losses
¿	variables
non_trainable_variables
 layer_regularization_losses
Àtrainable_variables
layer_metrics
layers
metrics
 
 
 
µ
Âregularization_losses
Ã	variables
non_trainable_variables
 layer_regularization_losses
Ätrainable_variables
layer_metrics
layers
metrics
 
 
 
µ
Æregularization_losses
Ç	variables
non_trainable_variables
 layer_regularization_losses
Ètrainable_variables
layer_metrics
layers
metrics
ZX
VARIABLE_VALUEconv2d/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE
 

Ê0
Ë1

Ê0
Ë1
µ
Ìregularization_losses
Í	variables
non_trainable_variables
 layer_regularization_losses
Îtrainable_variables
layer_metrics
layers
metrics
 
 
 
µ
Ðregularization_losses
Ñ	variables
non_trainable_variables
 layer_regularization_losses
Òtrainable_variables
 layer_metrics
¡layers
¢metrics
 
 
 
µ
Ôregularization_losses
Õ	variables
£non_trainable_variables
 ¤layer_regularization_losses
Ötrainable_variables
¥layer_metrics
¦layers
§metrics
YW
VARIABLE_VALUEdense/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUE
dense/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE
 

Ø0
Ù1

Ø0
Ù1
µ
Úregularization_losses
Û	variables
¨non_trainable_variables
 ©layer_regularization_losses
Ütrainable_variables
ªlayer_metrics
«layers
¬metrics
[Y
VARIABLE_VALUEdense_1/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_1/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE
 

Þ0
ß1

Þ0
ß1
µ
àregularization_losses
á	variables
­non_trainable_variables
 ®layer_regularization_losses
âtrainable_variables
¯layer_metrics
°layers
±metrics
[Y
VARIABLE_VALUEdense_2/kernel7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_2/bias5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE
 

ä0
å1

ä0
å1
µ
æregularization_losses
ç	variables
²non_trainable_variables
 ³layer_regularization_losses
ètrainable_variables
´layer_metrics
µlayers
¶metrics
[Y
VARIABLE_VALUEdense_3/kernel7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_3/bias5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUE
 

ê0
ë1

ê0
ë1
µ
ìregularization_losses
í	variables
·non_trainable_variables
 ¸layer_regularization_losses
îtrainable_variables
¹layer_metrics
ºlayers
»metrics
[Y
VARIABLE_VALUEoutput1/kernel7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEoutput1/bias5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUE
 

ð0
ñ1

ð0
ñ1
µ
òregularization_losses
ó	variables
¼non_trainable_variables
 ½layer_regularization_losses
ôtrainable_variables
¾layer_metrics
¿layers
Àmetrics
[Y
VARIABLE_VALUEoutput2/kernel7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEoutput2/bias5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUE
 

ö0
÷1

ö0
÷1
µ
øregularization_losses
ù	variables
Ánon_trainable_variables
 Âlayer_regularization_losses
útrainable_variables
Ãlayer_metrics
Älayers
Åmetrics
[Y
VARIABLE_VALUEoutput3/kernel7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEoutput3/bias5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUE
 

ü0
ý1

ü0
ý1
µ
þregularization_losses
ÿ	variables
Ænon_trainable_variables
 Çlayer_regularization_losses
trainable_variables
Èlayer_metrics
Élayers
Êmetrics
[Y
VARIABLE_VALUEoutput4/kernel7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEoutput4/bias5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
µ
regularization_losses
	variables
Ënon_trainable_variables
 Ìlayer_regularization_losses
trainable_variables
Ílayer_metrics
Îlayers
Ïmetrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
¦
40
51
62
<3
B4
C5
D6
E7
N8
T9
U10
V11
W12
h13
i14
n15
o16
x17
~18
19
20
21
22
23
24
25
26
 27
¡28
¦29
§30
°31
¶32
·33
¸34
¹35
 
 
¾
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
(
Ð0
Ñ1
Ò2
Ó3
Ô4
 
 
 
 
 

<0
 
 
 
 

B0
C1
D2
E3
 
 
 
 
 
 
 
 
 

N0
 
 
 
 

T0
U1
V2
W3
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

h0
i1
 
 
 
 

n0
o1
 
 
 
 
 
 
 
 
 

x0
 
 
 
 

~0
1
2
3
 
 
 
 

0
 
 
 
 
 
0
1
2
3
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

 0
¡1
 
 
 
 

¦0
§1
 
 
 
 
 
 
 
 
 

°0
 
 
 
 
 
¶0
·1
¸2
¹3
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

Õtotal

Öcount
×	variables
Ø	keras_api
8

Ùtotal

Úcount
Û	variables
Ü	keras_api
8

Ýtotal

Þcount
ß	variables
à	keras_api
8

átotal

âcount
ã	variables
ä	keras_api
8

åtotal

æcount
ç	variables
è	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

Õ0
Ö1

×	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE

Ù0
Ú1

Û	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE

Ý0
Þ1

ß	variables
QO
VARIABLE_VALUEtotal_34keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_34keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUE

á0
â1

ã	variables
QO
VARIABLE_VALUEtotal_44keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_44keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUE

å0
æ1

ç	variables
}{
VARIABLE_VALUEAdam/conv2d/kernel/mSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d/bias/mQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense/kernel/mSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense/bias/mQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_1/kernel/mSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_1/bias/mQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_2/kernel/mSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_2/bias/mQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_3/kernel/mSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_3/bias/mQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/output1/kernel/mSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/output1/bias/mQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/output2/kernel/mSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/output2/bias/mQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/output3/kernel/mSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/output3/bias/mQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/output4/kernel/mSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/output4/bias/mQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/conv2d/kernel/vSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d/bias/vQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense/kernel/vSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense/bias/vQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_1/kernel/vSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_1/bias/vQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_2/kernel/vSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_2/bias/vQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_3/kernel/vSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_3/bias/vQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/output1/kernel/vSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/output1/bias/vQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/output2/kernel/vSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/output2/bias/vQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/output3/kernel/vSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/output3/bias/vQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/output4/kernel/vSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/output4/bias/vQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_1Placeholder*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬*
dtype0*&
shape:ÿÿÿÿÿÿÿÿÿ¬¬
É
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1normalization/meannormalization/variancestem_conv/kernelstem_bn/gammastem_bn/betastem_bn/moving_meanstem_bn/moving_varianceblock1a_dwconv/depthwise_kernelblock1a_bn/gammablock1a_bn/betablock1a_bn/moving_meanblock1a_bn/moving_varianceblock1a_se_reduce/kernelblock1a_se_reduce/biasblock1a_se_expand/kernelblock1a_se_expand/biasblock1a_project_conv/kernelblock1a_project_bn/gammablock1a_project_bn/betablock1a_project_bn/moving_mean"block1a_project_bn/moving_varianceblock1b_dwconv/depthwise_kernelblock1b_bn/gammablock1b_bn/betablock1b_bn/moving_meanblock1b_bn/moving_varianceblock1b_se_reduce/kernelblock1b_se_reduce/biasblock1b_se_expand/kernelblock1b_se_expand/biasblock1b_project_conv/kernelblock1b_project_bn/gammablock1b_project_bn/betablock1b_project_bn/moving_mean"block1b_project_bn/moving_varianceconv2d/kernelconv2d/biasdense_3/kerneldense_3/biasdense_2/kerneldense_2/biasdense_1/kerneldense_1/biasdense/kernel
dense/biasoutput4/kerneloutput4/biasoutput3/kerneloutput3/biasoutput2/kerneloutput2/biasoutput1/kerneloutput1/bias*A
Tin:
826*
Tout
2*
_collective_manager_ids
 *`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*W
_read_only_resource_inputs9
75	
 !"#$%&'()*+,-./012345*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_100449
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
å#
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename&normalization/mean/Read/ReadVariableOp*normalization/variance/Read/ReadVariableOp'normalization/count/Read/ReadVariableOp$stem_conv/kernel/Read/ReadVariableOp!stem_bn/gamma/Read/ReadVariableOp stem_bn/beta/Read/ReadVariableOp'stem_bn/moving_mean/Read/ReadVariableOp+stem_bn/moving_variance/Read/ReadVariableOp3block1a_dwconv/depthwise_kernel/Read/ReadVariableOp$block1a_bn/gamma/Read/ReadVariableOp#block1a_bn/beta/Read/ReadVariableOp*block1a_bn/moving_mean/Read/ReadVariableOp.block1a_bn/moving_variance/Read/ReadVariableOp,block1a_se_reduce/kernel/Read/ReadVariableOp*block1a_se_reduce/bias/Read/ReadVariableOp,block1a_se_expand/kernel/Read/ReadVariableOp*block1a_se_expand/bias/Read/ReadVariableOp/block1a_project_conv/kernel/Read/ReadVariableOp,block1a_project_bn/gamma/Read/ReadVariableOp+block1a_project_bn/beta/Read/ReadVariableOp2block1a_project_bn/moving_mean/Read/ReadVariableOp6block1a_project_bn/moving_variance/Read/ReadVariableOp3block1b_dwconv/depthwise_kernel/Read/ReadVariableOp$block1b_bn/gamma/Read/ReadVariableOp#block1b_bn/beta/Read/ReadVariableOp*block1b_bn/moving_mean/Read/ReadVariableOp.block1b_bn/moving_variance/Read/ReadVariableOp,block1b_se_reduce/kernel/Read/ReadVariableOp*block1b_se_reduce/bias/Read/ReadVariableOp,block1b_se_expand/kernel/Read/ReadVariableOp*block1b_se_expand/bias/Read/ReadVariableOp/block1b_project_conv/kernel/Read/ReadVariableOp,block1b_project_bn/gamma/Read/ReadVariableOp+block1b_project_bn/beta/Read/ReadVariableOp2block1b_project_bn/moving_mean/Read/ReadVariableOp6block1b_project_bn/moving_variance/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp"output1/kernel/Read/ReadVariableOp output1/bias/Read/ReadVariableOp"output2/kernel/Read/ReadVariableOp output2/bias/Read/ReadVariableOp"output3/kernel/Read/ReadVariableOp output3/bias/Read/ReadVariableOp"output4/kernel/Read/ReadVariableOp output4/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_3/Read/ReadVariableOpcount_3/Read/ReadVariableOptotal_4/Read/ReadVariableOpcount_4/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp)Adam/dense_3/kernel/m/Read/ReadVariableOp'Adam/dense_3/bias/m/Read/ReadVariableOp)Adam/output1/kernel/m/Read/ReadVariableOp'Adam/output1/bias/m/Read/ReadVariableOp)Adam/output2/kernel/m/Read/ReadVariableOp'Adam/output2/bias/m/Read/ReadVariableOp)Adam/output3/kernel/m/Read/ReadVariableOp'Adam/output3/bias/m/Read/ReadVariableOp)Adam/output4/kernel/m/Read/ReadVariableOp'Adam/output4/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp)Adam/dense_3/kernel/v/Read/ReadVariableOp'Adam/dense_3/bias/v/Read/ReadVariableOp)Adam/output1/kernel/v/Read/ReadVariableOp'Adam/output1/bias/v/Read/ReadVariableOp)Adam/output2/kernel/v/Read/ReadVariableOp'Adam/output2/bias/v/Read/ReadVariableOp)Adam/output3/kernel/v/Read/ReadVariableOp'Adam/output3/bias/v/Read/ReadVariableOp)Adam/output4/kernel/v/Read/ReadVariableOp'Adam/output4/bias/v/Read/ReadVariableOpConst*v
Tino
m2k		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_102653
¬
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamenormalization/meannormalization/variancenormalization/countstem_conv/kernelstem_bn/gammastem_bn/betastem_bn/moving_meanstem_bn/moving_varianceblock1a_dwconv/depthwise_kernelblock1a_bn/gammablock1a_bn/betablock1a_bn/moving_meanblock1a_bn/moving_varianceblock1a_se_reduce/kernelblock1a_se_reduce/biasblock1a_se_expand/kernelblock1a_se_expand/biasblock1a_project_conv/kernelblock1a_project_bn/gammablock1a_project_bn/betablock1a_project_bn/moving_mean"block1a_project_bn/moving_varianceblock1b_dwconv/depthwise_kernelblock1b_bn/gammablock1b_bn/betablock1b_bn/moving_meanblock1b_bn/moving_varianceblock1b_se_reduce/kernelblock1b_se_reduce/biasblock1b_se_expand/kernelblock1b_se_expand/biasblock1b_project_conv/kernelblock1b_project_bn/gammablock1b_project_bn/betablock1b_project_bn/moving_mean"block1b_project_bn/moving_varianceconv2d/kernelconv2d/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasoutput1/kerneloutput1/biasoutput2/kerneloutput2/biasoutput3/kerneloutput3/biasoutput4/kerneloutput4/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1total_2count_2total_3count_3total_4count_4Adam/conv2d/kernel/mAdam/conv2d/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/dense_3/kernel/mAdam/dense_3/bias/mAdam/output1/kernel/mAdam/output1/bias/mAdam/output2/kernel/mAdam/output2/bias/mAdam/output3/kernel/mAdam/output3/bias/mAdam/output4/kernel/mAdam/output4/bias/mAdam/conv2d/kernel/vAdam/conv2d/bias/vAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/dense_2/kernel/vAdam/dense_2/bias/vAdam/dense_3/kernel/vAdam/dense_3/bias/vAdam/output1/kernel/vAdam/output1/bias/vAdam/output2/kernel/vAdam/output2/bias/vAdam/output3/kernel/vAdam/output3/bias/vAdam/output4/kernel/vAdam/output4/bias/v*u
Tinn
l2j*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_102978ÿ«"
ð	
Ü
C__inference_dense_1_layer_call_and_return_conditional_losses_102187

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ð	
Ü
C__inference_dense_2_layer_call_and_return_conditional_losses_102207

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ö
é
F__inference_block1b_bn_layer_call_and_return_conditional_losses_101744

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ì
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ü
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È

+__inference_block1b_bn_layer_call_fn_101775

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_block1b_bn_layer_call_and_return_conditional_losses_990252
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ï	
Û
B__inference_dense_1_layer_call_and_return_conditional_losses_99444

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


(__inference_stem_bn_layer_call_fn_101296

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¦
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_stem_bn_layer_call_and_return_conditional_losses_980992
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs

ç
L__inference_block1a_se_reduce_layer_call_and_return_conditional_losses_98857

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource

identity_1¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:(
*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
BiasAddi
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
Sigmoidj
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
mulc
IdentityIdentitymul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

IdentityÄ
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*+
_gradient_op_typeCustomGradient-98850*J
_output_shapes8
6:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
2
	IdentityN£

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity_1"!

identity_1Identity_1:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ(::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
¶
é
F__inference_block1a_bn_layer_call_and_return_conditional_losses_101391

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:(*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:(*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:(*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:(*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(:(:(:(:(:*
epsilon%o:*
is_training( 2
FusedBatchNormV3ì
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
§
I
-__inference_max_pooling2d_layer_call_fn_98596

inputs
identityé
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_985902
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
î	
Ú
A__inference_dense_layer_call_and_return_conditional_losses_102167

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ò
e
G__inference_block1b_drop_layer_call_and_return_conditional_losses_99309

inputs

identity_1d
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identitys

Identity_1IdentityIdentity:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³
i
M__inference_block1b_se_squeeze_layer_call_and_return_conditional_losses_98466

inputs
identity
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
þ
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_98590

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®
t
.__inference_block1a_dwconv_layer_call_fn_98128

inputs
unknown
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_block1a_dwconv_layer_call_and_return_conditional_losses_981202
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(:22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
à
v
L__inference_block1b_se_excite_layer_call_and_return_conditional_losses_99185

inputs
inputs_1
identity_
mulMulinputsinputs_1*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mule
IdentityIdentitymul:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:WS
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
è
E__inference_block1a_bn_layer_call_and_return_conditional_losses_98217

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:(*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:(*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:(*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:(*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(:(:(:(:(:*
epsilon%o:*
is_training( 2
FusedBatchNormV3ì
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs

Ñ
$__inference_signature_wrapper_100449
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51
identity

identity_1

identity_2

identity_3¢StatefulPartitionedCallÎ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51*A
Tin:
826*
Tout
2*
_collective_manager_ids
 *`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*W
_read_only_resource_inputs9
75	
 !"#$%&'()*+,-./012345*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_979972
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*
_input_shapesô
ñ:ÿÿÿÿÿÿÿÿÿ¬¬:::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬
!
_user_specified_name	input_1
¹
Ó
&__inference_model_layer_call_fn_100322
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51
identity

identity_1

identity_2

identity_3¢StatefulPartitionedCallï
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51*A
Tin:
826*
Tout
2*
_collective_manager_ids
 *`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*W
_read_only_resource_inputs9
75	
 !"#$%&'()*+,-./012345*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_1002072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*
_input_shapesô
ñ:ÿÿÿÿÿÿÿÿÿ¬¬:::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬
!
_user_specified_name	input_1
î
i
M__inference_block1a_se_reshape_layer_call_and_return_conditional_losses_98833

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :(2
Reshape/shape/3º
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
²
¾%
A__inference_model_layer_call_and_return_conditional_losses_100986

inputs1
-normalization_reshape_readvariableop_resource3
/normalization_reshape_1_readvariableop_resource,
(stem_conv_conv2d_readvariableop_resource#
stem_bn_readvariableop_resource%
!stem_bn_readvariableop_1_resource4
0stem_bn_fusedbatchnormv3_readvariableop_resource6
2stem_bn_fusedbatchnormv3_readvariableop_1_resource4
0block1a_dwconv_depthwise_readvariableop_resource&
"block1a_bn_readvariableop_resource(
$block1a_bn_readvariableop_1_resource7
3block1a_bn_fusedbatchnormv3_readvariableop_resource9
5block1a_bn_fusedbatchnormv3_readvariableop_1_resource4
0block1a_se_reduce_conv2d_readvariableop_resource5
1block1a_se_reduce_biasadd_readvariableop_resource4
0block1a_se_expand_conv2d_readvariableop_resource5
1block1a_se_expand_biasadd_readvariableop_resource7
3block1a_project_conv_conv2d_readvariableop_resource.
*block1a_project_bn_readvariableop_resource0
,block1a_project_bn_readvariableop_1_resource?
;block1a_project_bn_fusedbatchnormv3_readvariableop_resourceA
=block1a_project_bn_fusedbatchnormv3_readvariableop_1_resource4
0block1b_dwconv_depthwise_readvariableop_resource&
"block1b_bn_readvariableop_resource(
$block1b_bn_readvariableop_1_resource7
3block1b_bn_fusedbatchnormv3_readvariableop_resource9
5block1b_bn_fusedbatchnormv3_readvariableop_1_resource4
0block1b_se_reduce_conv2d_readvariableop_resource5
1block1b_se_reduce_biasadd_readvariableop_resource4
0block1b_se_expand_conv2d_readvariableop_resource5
1block1b_se_expand_biasadd_readvariableop_resource7
3block1b_project_conv_conv2d_readvariableop_resource.
*block1b_project_bn_readvariableop_resource0
,block1b_project_bn_readvariableop_1_resource?
;block1b_project_bn_fusedbatchnormv3_readvariableop_resourceA
=block1b_project_bn_fusedbatchnormv3_readvariableop_1_resource)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&output4_matmul_readvariableop_resource+
'output4_biasadd_readvariableop_resource*
&output3_matmul_readvariableop_resource+
'output3_biasadd_readvariableop_resource*
&output2_matmul_readvariableop_resource+
'output2_biasadd_readvariableop_resource*
&output1_matmul_readvariableop_resource+
'output1_biasadd_readvariableop_resource
identity

identity_1

identity_2

identity_3¢*block1a_bn/FusedBatchNormV3/ReadVariableOp¢,block1a_bn/FusedBatchNormV3/ReadVariableOp_1¢block1a_bn/ReadVariableOp¢block1a_bn/ReadVariableOp_1¢'block1a_dwconv/depthwise/ReadVariableOp¢2block1a_project_bn/FusedBatchNormV3/ReadVariableOp¢4block1a_project_bn/FusedBatchNormV3/ReadVariableOp_1¢!block1a_project_bn/ReadVariableOp¢#block1a_project_bn/ReadVariableOp_1¢*block1a_project_conv/Conv2D/ReadVariableOp¢(block1a_se_expand/BiasAdd/ReadVariableOp¢'block1a_se_expand/Conv2D/ReadVariableOp¢(block1a_se_reduce/BiasAdd/ReadVariableOp¢'block1a_se_reduce/Conv2D/ReadVariableOp¢*block1b_bn/FusedBatchNormV3/ReadVariableOp¢,block1b_bn/FusedBatchNormV3/ReadVariableOp_1¢block1b_bn/ReadVariableOp¢block1b_bn/ReadVariableOp_1¢'block1b_dwconv/depthwise/ReadVariableOp¢2block1b_project_bn/FusedBatchNormV3/ReadVariableOp¢4block1b_project_bn/FusedBatchNormV3/ReadVariableOp_1¢!block1b_project_bn/ReadVariableOp¢#block1b_project_bn/ReadVariableOp_1¢*block1b_project_conv/Conv2D/ReadVariableOp¢(block1b_se_expand/BiasAdd/ReadVariableOp¢'block1b_se_expand/Conv2D/ReadVariableOp¢(block1b_se_reduce/BiasAdd/ReadVariableOp¢'block1b_se_reduce/Conv2D/ReadVariableOp¢conv2d/BiasAdd/ReadVariableOp¢conv2d/Conv2D/ReadVariableOp¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOp¢dense_3/BiasAdd/ReadVariableOp¢dense_3/MatMul/ReadVariableOp¢$normalization/Reshape/ReadVariableOp¢&normalization/Reshape_1/ReadVariableOp¢output1/BiasAdd/ReadVariableOp¢output1/MatMul/ReadVariableOp¢output2/BiasAdd/ReadVariableOp¢output2/MatMul/ReadVariableOp¢output3/BiasAdd/ReadVariableOp¢output3/MatMul/ReadVariableOp¢output4/BiasAdd/ReadVariableOp¢output4/MatMul/ReadVariableOp¢'stem_bn/FusedBatchNormV3/ReadVariableOp¢)stem_bn/FusedBatchNormV3/ReadVariableOp_1¢stem_bn/ReadVariableOp¢stem_bn/ReadVariableOp_1¢stem_conv/Conv2D/ReadVariableOpi
rescaling/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;2
rescaling/Cast/xm
rescaling/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rescaling/Cast_1/x
rescaling/mulMulinputsrescaling/Cast/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬2
rescaling/mul
rescaling/addAddV2rescaling/mul:z:0rescaling/Cast_1/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬2
rescaling/add¶
$normalization/Reshape/ReadVariableOpReadVariableOp-normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02&
$normalization/Reshape/ReadVariableOp
normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
normalization/Reshape/shape¾
normalization/ReshapeReshape,normalization/Reshape/ReadVariableOp:value:0$normalization/Reshape/shape:output:0*
T0*&
_output_shapes
:2
normalization/Reshape¼
&normalization/Reshape_1/ReadVariableOpReadVariableOp/normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization/Reshape_1/ReadVariableOp
normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
normalization/Reshape_1/shapeÆ
normalization/Reshape_1Reshape.normalization/Reshape_1/ReadVariableOp:value:0&normalization/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
normalization/Reshape_1
normalization/subSubrescaling/add:z:0normalization/Reshape:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬2
normalization/sub
normalization/SqrtSqrt normalization/Reshape_1:output:0*
T0*&
_output_shapes
:2
normalization/Sqrtw
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö32
normalization/Maximum/y¤
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*&
_output_shapes
:2
normalization/Maximum§
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬2
normalization/truediv©
stem_conv_pad/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                               2
stem_conv_pad/Pad/paddings©
stem_conv_pad/PadPadnormalization/truediv:z:0#stem_conv_pad/Pad/paddings:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­­2
stem_conv_pad/Pad³
stem_conv/Conv2D/ReadVariableOpReadVariableOp(stem_conv_conv2d_readvariableop_resource*&
_output_shapes
:(*
dtype02!
stem_conv/Conv2D/ReadVariableOpØ
stem_conv/Conv2DConv2Dstem_conv_pad/Pad:output:0'stem_conv/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
paddingVALID*
strides
2
stem_conv/Conv2D
stem_bn/ReadVariableOpReadVariableOpstem_bn_readvariableop_resource*
_output_shapes
:(*
dtype02
stem_bn/ReadVariableOp
stem_bn/ReadVariableOp_1ReadVariableOp!stem_bn_readvariableop_1_resource*
_output_shapes
:(*
dtype02
stem_bn/ReadVariableOp_1¿
'stem_bn/FusedBatchNormV3/ReadVariableOpReadVariableOp0stem_bn_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:(*
dtype02)
'stem_bn/FusedBatchNormV3/ReadVariableOpÅ
)stem_bn/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp2stem_bn_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:(*
dtype02+
)stem_bn/FusedBatchNormV3/ReadVariableOp_1
stem_bn/FusedBatchNormV3FusedBatchNormV3stem_conv/Conv2D:output:0stem_bn/ReadVariableOp:value:0 stem_bn/ReadVariableOp_1:value:0/stem_bn/FusedBatchNormV3/ReadVariableOp:value:01stem_bn/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ(:(:(:(:(:*
epsilon%o:*
is_training( 2
stem_bn/FusedBatchNormV3
stem_activation/SigmoidSigmoidstem_bn/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
stem_activation/Sigmoid¨
stem_activation/mulMulstem_bn/FusedBatchNormV3:y:0stem_activation/Sigmoid:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
stem_activation/mul
stem_activation/IdentityIdentitystem_activation/mul:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
stem_activation/Identity
stem_activation/IdentityN	IdentityNstem_activation/mul:z:0stem_bn/FusedBatchNormV3:y:0*
T
2*,
_gradient_op_typeCustomGradient-100764*N
_output_shapes<
::ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(2
stem_activation/IdentityNË
'block1a_dwconv/depthwise/ReadVariableOpReadVariableOp0block1a_dwconv_depthwise_readvariableop_resource*&
_output_shapes
:(*
dtype02)
'block1a_dwconv/depthwise/ReadVariableOp
block1a_dwconv/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      (      2 
block1a_dwconv/depthwise/Shape¡
&block1a_dwconv/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2(
&block1a_dwconv/depthwise/dilation_rate
block1a_dwconv/depthwiseDepthwiseConv2dNative"stem_activation/IdentityN:output:0/block1a_dwconv/depthwise/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
paddingSAME*
strides
2
block1a_dwconv/depthwise
block1a_bn/ReadVariableOpReadVariableOp"block1a_bn_readvariableop_resource*
_output_shapes
:(*
dtype02
block1a_bn/ReadVariableOp
block1a_bn/ReadVariableOp_1ReadVariableOp$block1a_bn_readvariableop_1_resource*
_output_shapes
:(*
dtype02
block1a_bn/ReadVariableOp_1È
*block1a_bn/FusedBatchNormV3/ReadVariableOpReadVariableOp3block1a_bn_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:(*
dtype02,
*block1a_bn/FusedBatchNormV3/ReadVariableOpÎ
,block1a_bn/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5block1a_bn_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:(*
dtype02.
,block1a_bn/FusedBatchNormV3/ReadVariableOp_1©
block1a_bn/FusedBatchNormV3FusedBatchNormV3!block1a_dwconv/depthwise:output:0!block1a_bn/ReadVariableOp:value:0#block1a_bn/ReadVariableOp_1:value:02block1a_bn/FusedBatchNormV3/ReadVariableOp:value:04block1a_bn/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ(:(:(:(:(:*
epsilon%o:*
is_training( 2
block1a_bn/FusedBatchNormV3 
block1a_activation/SigmoidSigmoidblock1a_bn/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
block1a_activation/Sigmoid´
block1a_activation/mulMulblock1a_bn/FusedBatchNormV3:y:0block1a_activation/Sigmoid:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
block1a_activation/mul
block1a_activation/IdentityIdentityblock1a_activation/mul:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
block1a_activation/Identity
block1a_activation/IdentityN	IdentityNblock1a_activation/mul:z:0block1a_bn/FusedBatchNormV3:y:0*
T
2*,
_gradient_op_typeCustomGradient-100789*N
_output_shapes<
::ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(2
block1a_activation/IdentityN§
)block1a_se_squeeze/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2+
)block1a_se_squeeze/Mean/reduction_indicesÇ
block1a_se_squeeze/MeanMean%block1a_activation/IdentityN:output:02block1a_se_squeeze/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
block1a_se_squeeze/Mean
block1a_se_reshape/ShapeShape block1a_se_squeeze/Mean:output:0*
T0*
_output_shapes
:2
block1a_se_reshape/Shape
&block1a_se_reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&block1a_se_reshape/strided_slice/stack
(block1a_se_reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(block1a_se_reshape/strided_slice/stack_1
(block1a_se_reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(block1a_se_reshape/strided_slice/stack_2Ô
 block1a_se_reshape/strided_sliceStridedSlice!block1a_se_reshape/Shape:output:0/block1a_se_reshape/strided_slice/stack:output:01block1a_se_reshape/strided_slice/stack_1:output:01block1a_se_reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 block1a_se_reshape/strided_slice
"block1a_se_reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"block1a_se_reshape/Reshape/shape/1
"block1a_se_reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"block1a_se_reshape/Reshape/shape/2
"block1a_se_reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :(2$
"block1a_se_reshape/Reshape/shape/3¬
 block1a_se_reshape/Reshape/shapePack)block1a_se_reshape/strided_slice:output:0+block1a_se_reshape/Reshape/shape/1:output:0+block1a_se_reshape/Reshape/shape/2:output:0+block1a_se_reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2"
 block1a_se_reshape/Reshape/shapeÊ
block1a_se_reshape/ReshapeReshape block1a_se_squeeze/Mean:output:0)block1a_se_reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
block1a_se_reshape/ReshapeË
'block1a_se_reduce/Conv2D/ReadVariableOpReadVariableOp0block1a_se_reduce_conv2d_readvariableop_resource*&
_output_shapes
:(
*
dtype02)
'block1a_se_reduce/Conv2D/ReadVariableOpö
block1a_se_reduce/Conv2DConv2D#block1a_se_reshape/Reshape:output:0/block1a_se_reduce/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
paddingSAME*
strides
2
block1a_se_reduce/Conv2DÂ
(block1a_se_reduce/BiasAdd/ReadVariableOpReadVariableOp1block1a_se_reduce_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02*
(block1a_se_reduce/BiasAdd/ReadVariableOpÐ
block1a_se_reduce/BiasAddBiasAdd!block1a_se_reduce/Conv2D:output:00block1a_se_reduce/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
block1a_se_reduce/BiasAdd
block1a_se_reduce/SigmoidSigmoid"block1a_se_reduce/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
block1a_se_reduce/Sigmoid²
block1a_se_reduce/mulMul"block1a_se_reduce/BiasAdd:output:0block1a_se_reduce/Sigmoid:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
block1a_se_reduce/mul
block1a_se_reduce/IdentityIdentityblock1a_se_reduce/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
block1a_se_reduce/Identity
block1a_se_reduce/IdentityN	IdentityNblock1a_se_reduce/mul:z:0"block1a_se_reduce/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-100813*J
_output_shapes8
6:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
2
block1a_se_reduce/IdentityNË
'block1a_se_expand/Conv2D/ReadVariableOpReadVariableOp0block1a_se_expand_conv2d_readvariableop_resource*&
_output_shapes
:
(*
dtype02)
'block1a_se_expand/Conv2D/ReadVariableOp÷
block1a_se_expand/Conv2DConv2D$block1a_se_reduce/IdentityN:output:0/block1a_se_expand/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
paddingSAME*
strides
2
block1a_se_expand/Conv2DÂ
(block1a_se_expand/BiasAdd/ReadVariableOpReadVariableOp1block1a_se_expand_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02*
(block1a_se_expand/BiasAdd/ReadVariableOpÐ
block1a_se_expand/BiasAddBiasAdd!block1a_se_expand/Conv2D:output:00block1a_se_expand/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
block1a_se_expand/BiasAdd
block1a_se_expand/SigmoidSigmoid"block1a_se_expand/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
block1a_se_expand/Sigmoid·
block1a_se_excite/mulMul%block1a_activation/IdentityN:output:0block1a_se_expand/Sigmoid:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
block1a_se_excite/mulÔ
*block1a_project_conv/Conv2D/ReadVariableOpReadVariableOp3block1a_project_conv_conv2d_readvariableop_resource*&
_output_shapes
:(*
dtype02,
*block1a_project_conv/Conv2D/ReadVariableOp÷
block1a_project_conv/Conv2DConv2Dblock1a_se_excite/mul:z:02block1a_project_conv/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
block1a_project_conv/Conv2D­
!block1a_project_bn/ReadVariableOpReadVariableOp*block1a_project_bn_readvariableop_resource*
_output_shapes
:*
dtype02#
!block1a_project_bn/ReadVariableOp³
#block1a_project_bn/ReadVariableOp_1ReadVariableOp,block1a_project_bn_readvariableop_1_resource*
_output_shapes
:*
dtype02%
#block1a_project_bn/ReadVariableOp_1à
2block1a_project_bn/FusedBatchNormV3/ReadVariableOpReadVariableOp;block1a_project_bn_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype024
2block1a_project_bn/FusedBatchNormV3/ReadVariableOpæ
4block1a_project_bn/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp=block1a_project_bn_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype026
4block1a_project_bn/FusedBatchNormV3/ReadVariableOp_1Ü
#block1a_project_bn/FusedBatchNormV3FusedBatchNormV3$block1a_project_conv/Conv2D:output:0)block1a_project_bn/ReadVariableOp:value:0+block1a_project_bn/ReadVariableOp_1:value:0:block1a_project_bn/FusedBatchNormV3/ReadVariableOp:value:0<block1a_project_bn/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2%
#block1a_project_bn/FusedBatchNormV3Ë
'block1b_dwconv/depthwise/ReadVariableOpReadVariableOp0block1b_dwconv_depthwise_readvariableop_resource*&
_output_shapes
:*
dtype02)
'block1b_dwconv/depthwise/ReadVariableOp
block1b_dwconv/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2 
block1b_dwconv/depthwise/Shape¡
&block1b_dwconv/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2(
&block1b_dwconv/depthwise/dilation_rate
block1b_dwconv/depthwiseDepthwiseConv2dNative'block1a_project_bn/FusedBatchNormV3:y:0/block1b_dwconv/depthwise/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
block1b_dwconv/depthwise
block1b_bn/ReadVariableOpReadVariableOp"block1b_bn_readvariableop_resource*
_output_shapes
:*
dtype02
block1b_bn/ReadVariableOp
block1b_bn/ReadVariableOp_1ReadVariableOp$block1b_bn_readvariableop_1_resource*
_output_shapes
:*
dtype02
block1b_bn/ReadVariableOp_1È
*block1b_bn/FusedBatchNormV3/ReadVariableOpReadVariableOp3block1b_bn_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*block1b_bn/FusedBatchNormV3/ReadVariableOpÎ
,block1b_bn/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5block1b_bn_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,block1b_bn/FusedBatchNormV3/ReadVariableOp_1©
block1b_bn/FusedBatchNormV3FusedBatchNormV3!block1b_dwconv/depthwise:output:0!block1b_bn/ReadVariableOp:value:0#block1b_bn/ReadVariableOp_1:value:02block1b_bn/FusedBatchNormV3/ReadVariableOp:value:04block1b_bn/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
block1b_bn/FusedBatchNormV3 
block1b_activation/SigmoidSigmoidblock1b_bn/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
block1b_activation/Sigmoid´
block1b_activation/mulMulblock1b_bn/FusedBatchNormV3:y:0block1b_activation/Sigmoid:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
block1b_activation/mul
block1b_activation/IdentityIdentityblock1b_activation/mul:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
block1b_activation/Identity
block1b_activation/IdentityN	IdentityNblock1b_activation/mul:z:0block1b_bn/FusedBatchNormV3:y:0*
T
2*,
_gradient_op_typeCustomGradient-100863*N
_output_shapes<
::ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ2
block1b_activation/IdentityN§
)block1b_se_squeeze/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2+
)block1b_se_squeeze/Mean/reduction_indicesÇ
block1b_se_squeeze/MeanMean%block1b_activation/IdentityN:output:02block1b_se_squeeze/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
block1b_se_squeeze/Mean
block1b_se_reshape/ShapeShape block1b_se_squeeze/Mean:output:0*
T0*
_output_shapes
:2
block1b_se_reshape/Shape
&block1b_se_reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&block1b_se_reshape/strided_slice/stack
(block1b_se_reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(block1b_se_reshape/strided_slice/stack_1
(block1b_se_reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(block1b_se_reshape/strided_slice/stack_2Ô
 block1b_se_reshape/strided_sliceStridedSlice!block1b_se_reshape/Shape:output:0/block1b_se_reshape/strided_slice/stack:output:01block1b_se_reshape/strided_slice/stack_1:output:01block1b_se_reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 block1b_se_reshape/strided_slice
"block1b_se_reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"block1b_se_reshape/Reshape/shape/1
"block1b_se_reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"block1b_se_reshape/Reshape/shape/2
"block1b_se_reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2$
"block1b_se_reshape/Reshape/shape/3¬
 block1b_se_reshape/Reshape/shapePack)block1b_se_reshape/strided_slice:output:0+block1b_se_reshape/Reshape/shape/1:output:0+block1b_se_reshape/Reshape/shape/2:output:0+block1b_se_reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2"
 block1b_se_reshape/Reshape/shapeÊ
block1b_se_reshape/ReshapeReshape block1b_se_squeeze/Mean:output:0)block1b_se_reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
block1b_se_reshape/ReshapeË
'block1b_se_reduce/Conv2D/ReadVariableOpReadVariableOp0block1b_se_reduce_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'block1b_se_reduce/Conv2D/ReadVariableOpö
block1b_se_reduce/Conv2DConv2D#block1b_se_reshape/Reshape:output:0/block1b_se_reduce/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
block1b_se_reduce/Conv2DÂ
(block1b_se_reduce/BiasAdd/ReadVariableOpReadVariableOp1block1b_se_reduce_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(block1b_se_reduce/BiasAdd/ReadVariableOpÐ
block1b_se_reduce/BiasAddBiasAdd!block1b_se_reduce/Conv2D:output:00block1b_se_reduce/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
block1b_se_reduce/BiasAdd
block1b_se_reduce/SigmoidSigmoid"block1b_se_reduce/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
block1b_se_reduce/Sigmoid²
block1b_se_reduce/mulMul"block1b_se_reduce/BiasAdd:output:0block1b_se_reduce/Sigmoid:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
block1b_se_reduce/mul
block1b_se_reduce/IdentityIdentityblock1b_se_reduce/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
block1b_se_reduce/Identity
block1b_se_reduce/IdentityN	IdentityNblock1b_se_reduce/mul:z:0"block1b_se_reduce/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-100887*J
_output_shapes8
6:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ2
block1b_se_reduce/IdentityNË
'block1b_se_expand/Conv2D/ReadVariableOpReadVariableOp0block1b_se_expand_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'block1b_se_expand/Conv2D/ReadVariableOp÷
block1b_se_expand/Conv2DConv2D$block1b_se_reduce/IdentityN:output:0/block1b_se_expand/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
block1b_se_expand/Conv2DÂ
(block1b_se_expand/BiasAdd/ReadVariableOpReadVariableOp1block1b_se_expand_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(block1b_se_expand/BiasAdd/ReadVariableOpÐ
block1b_se_expand/BiasAddBiasAdd!block1b_se_expand/Conv2D:output:00block1b_se_expand/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
block1b_se_expand/BiasAdd
block1b_se_expand/SigmoidSigmoid"block1b_se_expand/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
block1b_se_expand/Sigmoid·
block1b_se_excite/mulMul%block1b_activation/IdentityN:output:0block1b_se_expand/Sigmoid:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
block1b_se_excite/mulÔ
*block1b_project_conv/Conv2D/ReadVariableOpReadVariableOp3block1b_project_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02,
*block1b_project_conv/Conv2D/ReadVariableOp÷
block1b_project_conv/Conv2DConv2Dblock1b_se_excite/mul:z:02block1b_project_conv/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
block1b_project_conv/Conv2D­
!block1b_project_bn/ReadVariableOpReadVariableOp*block1b_project_bn_readvariableop_resource*
_output_shapes
:*
dtype02#
!block1b_project_bn/ReadVariableOp³
#block1b_project_bn/ReadVariableOp_1ReadVariableOp,block1b_project_bn_readvariableop_1_resource*
_output_shapes
:*
dtype02%
#block1b_project_bn/ReadVariableOp_1à
2block1b_project_bn/FusedBatchNormV3/ReadVariableOpReadVariableOp;block1b_project_bn_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype024
2block1b_project_bn/FusedBatchNormV3/ReadVariableOpæ
4block1b_project_bn/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp=block1b_project_bn_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype026
4block1b_project_bn/FusedBatchNormV3/ReadVariableOp_1Ü
#block1b_project_bn/FusedBatchNormV3FusedBatchNormV3$block1b_project_conv/Conv2D:output:0)block1b_project_bn/ReadVariableOp:value:0+block1b_project_bn/ReadVariableOp_1:value:0:block1b_project_bn/FusedBatchNormV3/ReadVariableOp:value:0<block1b_project_bn/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2%
#block1b_project_bn/FusedBatchNormV3
block1b_drop/IdentityIdentity'block1b_project_bn/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
block1b_drop/Identity°
block1b_add/addAddV2block1b_drop/Identity:output:0'block1a_project_bn/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
block1b_add/addÌ
average_pooling2d/AvgPoolAvgPoolblock1b_add/add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2
average_pooling2d/AvgPoolª
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOpÕ
conv2d/Conv2DConv2D"average_pooling2d/AvgPool:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d/Conv2D¡
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp¤
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d/BiasAdd¿
max_pooling2d/MaxPoolMaxPoolconv2d/BiasAdd:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
  2
flatten/Const
flatten/ReshapeReshapemax_pooling2d/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten/Reshape¦
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense_3/MatMul/ReadVariableOp
dense_3/MatMulMatMulflatten/Reshape:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_3/MatMul¤
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp¡
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_3/BiasAddp
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_3/Relu¦
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMulflatten/Reshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/MatMul¤
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp¡
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/Relu¦
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/MatMul¤
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp¡
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/Relu 
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

dense/Relu¥
output4/MatMul/ReadVariableOpReadVariableOp&output4_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
output4/MatMul/ReadVariableOp
output4/MatMulMatMuldense_3/Relu:activations:0%output4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
output4/MatMul¤
output4/BiasAdd/ReadVariableOpReadVariableOp'output4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
output4/BiasAdd/ReadVariableOp¡
output4/BiasAddBiasAddoutput4/MatMul:product:0&output4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
output4/BiasAdd¥
output3/MatMul/ReadVariableOpReadVariableOp&output3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
output3/MatMul/ReadVariableOp
output3/MatMulMatMuldense_2/Relu:activations:0%output3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
output3/MatMul¤
output3/BiasAdd/ReadVariableOpReadVariableOp'output3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
output3/BiasAdd/ReadVariableOp¡
output3/BiasAddBiasAddoutput3/MatMul:product:0&output3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
output3/BiasAdd¥
output2/MatMul/ReadVariableOpReadVariableOp&output2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
output2/MatMul/ReadVariableOp
output2/MatMulMatMuldense_1/Relu:activations:0%output2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
output2/MatMul¤
output2/BiasAdd/ReadVariableOpReadVariableOp'output2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
output2/BiasAdd/ReadVariableOp¡
output2/BiasAddBiasAddoutput2/MatMul:product:0&output2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
output2/BiasAdd¥
output1/MatMul/ReadVariableOpReadVariableOp&output1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
output1/MatMul/ReadVariableOp
output1/MatMulMatMuldense/Relu:activations:0%output1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
output1/MatMul¤
output1/BiasAdd/ReadVariableOpReadVariableOp'output1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
output1/BiasAdd/ReadVariableOp¡
output1/BiasAddBiasAddoutput1/MatMul:product:0&output1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
output1/BiasAddÅ
IdentityIdentityoutput1/BiasAdd:output:0+^block1a_bn/FusedBatchNormV3/ReadVariableOp-^block1a_bn/FusedBatchNormV3/ReadVariableOp_1^block1a_bn/ReadVariableOp^block1a_bn/ReadVariableOp_1(^block1a_dwconv/depthwise/ReadVariableOp3^block1a_project_bn/FusedBatchNormV3/ReadVariableOp5^block1a_project_bn/FusedBatchNormV3/ReadVariableOp_1"^block1a_project_bn/ReadVariableOp$^block1a_project_bn/ReadVariableOp_1+^block1a_project_conv/Conv2D/ReadVariableOp)^block1a_se_expand/BiasAdd/ReadVariableOp(^block1a_se_expand/Conv2D/ReadVariableOp)^block1a_se_reduce/BiasAdd/ReadVariableOp(^block1a_se_reduce/Conv2D/ReadVariableOp+^block1b_bn/FusedBatchNormV3/ReadVariableOp-^block1b_bn/FusedBatchNormV3/ReadVariableOp_1^block1b_bn/ReadVariableOp^block1b_bn/ReadVariableOp_1(^block1b_dwconv/depthwise/ReadVariableOp3^block1b_project_bn/FusedBatchNormV3/ReadVariableOp5^block1b_project_bn/FusedBatchNormV3/ReadVariableOp_1"^block1b_project_bn/ReadVariableOp$^block1b_project_bn/ReadVariableOp_1+^block1b_project_conv/Conv2D/ReadVariableOp)^block1b_se_expand/BiasAdd/ReadVariableOp(^block1b_se_expand/Conv2D/ReadVariableOp)^block1b_se_reduce/BiasAdd/ReadVariableOp(^block1b_se_reduce/Conv2D/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp^output1/BiasAdd/ReadVariableOp^output1/MatMul/ReadVariableOp^output2/BiasAdd/ReadVariableOp^output2/MatMul/ReadVariableOp^output3/BiasAdd/ReadVariableOp^output3/MatMul/ReadVariableOp^output4/BiasAdd/ReadVariableOp^output4/MatMul/ReadVariableOp(^stem_bn/FusedBatchNormV3/ReadVariableOp*^stem_bn/FusedBatchNormV3/ReadVariableOp_1^stem_bn/ReadVariableOp^stem_bn/ReadVariableOp_1 ^stem_conv/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÉ

Identity_1Identityoutput2/BiasAdd:output:0+^block1a_bn/FusedBatchNormV3/ReadVariableOp-^block1a_bn/FusedBatchNormV3/ReadVariableOp_1^block1a_bn/ReadVariableOp^block1a_bn/ReadVariableOp_1(^block1a_dwconv/depthwise/ReadVariableOp3^block1a_project_bn/FusedBatchNormV3/ReadVariableOp5^block1a_project_bn/FusedBatchNormV3/ReadVariableOp_1"^block1a_project_bn/ReadVariableOp$^block1a_project_bn/ReadVariableOp_1+^block1a_project_conv/Conv2D/ReadVariableOp)^block1a_se_expand/BiasAdd/ReadVariableOp(^block1a_se_expand/Conv2D/ReadVariableOp)^block1a_se_reduce/BiasAdd/ReadVariableOp(^block1a_se_reduce/Conv2D/ReadVariableOp+^block1b_bn/FusedBatchNormV3/ReadVariableOp-^block1b_bn/FusedBatchNormV3/ReadVariableOp_1^block1b_bn/ReadVariableOp^block1b_bn/ReadVariableOp_1(^block1b_dwconv/depthwise/ReadVariableOp3^block1b_project_bn/FusedBatchNormV3/ReadVariableOp5^block1b_project_bn/FusedBatchNormV3/ReadVariableOp_1"^block1b_project_bn/ReadVariableOp$^block1b_project_bn/ReadVariableOp_1+^block1b_project_conv/Conv2D/ReadVariableOp)^block1b_se_expand/BiasAdd/ReadVariableOp(^block1b_se_expand/Conv2D/ReadVariableOp)^block1b_se_reduce/BiasAdd/ReadVariableOp(^block1b_se_reduce/Conv2D/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp^output1/BiasAdd/ReadVariableOp^output1/MatMul/ReadVariableOp^output2/BiasAdd/ReadVariableOp^output2/MatMul/ReadVariableOp^output3/BiasAdd/ReadVariableOp^output3/MatMul/ReadVariableOp^output4/BiasAdd/ReadVariableOp^output4/MatMul/ReadVariableOp(^stem_bn/FusedBatchNormV3/ReadVariableOp*^stem_bn/FusedBatchNormV3/ReadVariableOp_1^stem_bn/ReadVariableOp^stem_bn/ReadVariableOp_1 ^stem_conv/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1É

Identity_2Identityoutput3/BiasAdd:output:0+^block1a_bn/FusedBatchNormV3/ReadVariableOp-^block1a_bn/FusedBatchNormV3/ReadVariableOp_1^block1a_bn/ReadVariableOp^block1a_bn/ReadVariableOp_1(^block1a_dwconv/depthwise/ReadVariableOp3^block1a_project_bn/FusedBatchNormV3/ReadVariableOp5^block1a_project_bn/FusedBatchNormV3/ReadVariableOp_1"^block1a_project_bn/ReadVariableOp$^block1a_project_bn/ReadVariableOp_1+^block1a_project_conv/Conv2D/ReadVariableOp)^block1a_se_expand/BiasAdd/ReadVariableOp(^block1a_se_expand/Conv2D/ReadVariableOp)^block1a_se_reduce/BiasAdd/ReadVariableOp(^block1a_se_reduce/Conv2D/ReadVariableOp+^block1b_bn/FusedBatchNormV3/ReadVariableOp-^block1b_bn/FusedBatchNormV3/ReadVariableOp_1^block1b_bn/ReadVariableOp^block1b_bn/ReadVariableOp_1(^block1b_dwconv/depthwise/ReadVariableOp3^block1b_project_bn/FusedBatchNormV3/ReadVariableOp5^block1b_project_bn/FusedBatchNormV3/ReadVariableOp_1"^block1b_project_bn/ReadVariableOp$^block1b_project_bn/ReadVariableOp_1+^block1b_project_conv/Conv2D/ReadVariableOp)^block1b_se_expand/BiasAdd/ReadVariableOp(^block1b_se_expand/Conv2D/ReadVariableOp)^block1b_se_reduce/BiasAdd/ReadVariableOp(^block1b_se_reduce/Conv2D/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp^output1/BiasAdd/ReadVariableOp^output1/MatMul/ReadVariableOp^output2/BiasAdd/ReadVariableOp^output2/MatMul/ReadVariableOp^output3/BiasAdd/ReadVariableOp^output3/MatMul/ReadVariableOp^output4/BiasAdd/ReadVariableOp^output4/MatMul/ReadVariableOp(^stem_bn/FusedBatchNormV3/ReadVariableOp*^stem_bn/FusedBatchNormV3/ReadVariableOp_1^stem_bn/ReadVariableOp^stem_bn/ReadVariableOp_1 ^stem_conv/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2É

Identity_3Identityoutput4/BiasAdd:output:0+^block1a_bn/FusedBatchNormV3/ReadVariableOp-^block1a_bn/FusedBatchNormV3/ReadVariableOp_1^block1a_bn/ReadVariableOp^block1a_bn/ReadVariableOp_1(^block1a_dwconv/depthwise/ReadVariableOp3^block1a_project_bn/FusedBatchNormV3/ReadVariableOp5^block1a_project_bn/FusedBatchNormV3/ReadVariableOp_1"^block1a_project_bn/ReadVariableOp$^block1a_project_bn/ReadVariableOp_1+^block1a_project_conv/Conv2D/ReadVariableOp)^block1a_se_expand/BiasAdd/ReadVariableOp(^block1a_se_expand/Conv2D/ReadVariableOp)^block1a_se_reduce/BiasAdd/ReadVariableOp(^block1a_se_reduce/Conv2D/ReadVariableOp+^block1b_bn/FusedBatchNormV3/ReadVariableOp-^block1b_bn/FusedBatchNormV3/ReadVariableOp_1^block1b_bn/ReadVariableOp^block1b_bn/ReadVariableOp_1(^block1b_dwconv/depthwise/ReadVariableOp3^block1b_project_bn/FusedBatchNormV3/ReadVariableOp5^block1b_project_bn/FusedBatchNormV3/ReadVariableOp_1"^block1b_project_bn/ReadVariableOp$^block1b_project_bn/ReadVariableOp_1+^block1b_project_conv/Conv2D/ReadVariableOp)^block1b_se_expand/BiasAdd/ReadVariableOp(^block1b_se_expand/Conv2D/ReadVariableOp)^block1b_se_reduce/BiasAdd/ReadVariableOp(^block1b_se_reduce/Conv2D/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp^output1/BiasAdd/ReadVariableOp^output1/MatMul/ReadVariableOp^output2/BiasAdd/ReadVariableOp^output2/MatMul/ReadVariableOp^output3/BiasAdd/ReadVariableOp^output3/MatMul/ReadVariableOp^output4/BiasAdd/ReadVariableOp^output4/MatMul/ReadVariableOp(^stem_bn/FusedBatchNormV3/ReadVariableOp*^stem_bn/FusedBatchNormV3/ReadVariableOp_1^stem_bn/ReadVariableOp^stem_bn/ReadVariableOp_1 ^stem_conv/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*
_input_shapesô
ñ:ÿÿÿÿÿÿÿÿÿ¬¬:::::::::::::::::::::::::::::::::::::::::::::::::::::2X
*block1a_bn/FusedBatchNormV3/ReadVariableOp*block1a_bn/FusedBatchNormV3/ReadVariableOp2\
,block1a_bn/FusedBatchNormV3/ReadVariableOp_1,block1a_bn/FusedBatchNormV3/ReadVariableOp_126
block1a_bn/ReadVariableOpblock1a_bn/ReadVariableOp2:
block1a_bn/ReadVariableOp_1block1a_bn/ReadVariableOp_12R
'block1a_dwconv/depthwise/ReadVariableOp'block1a_dwconv/depthwise/ReadVariableOp2h
2block1a_project_bn/FusedBatchNormV3/ReadVariableOp2block1a_project_bn/FusedBatchNormV3/ReadVariableOp2l
4block1a_project_bn/FusedBatchNormV3/ReadVariableOp_14block1a_project_bn/FusedBatchNormV3/ReadVariableOp_12F
!block1a_project_bn/ReadVariableOp!block1a_project_bn/ReadVariableOp2J
#block1a_project_bn/ReadVariableOp_1#block1a_project_bn/ReadVariableOp_12X
*block1a_project_conv/Conv2D/ReadVariableOp*block1a_project_conv/Conv2D/ReadVariableOp2T
(block1a_se_expand/BiasAdd/ReadVariableOp(block1a_se_expand/BiasAdd/ReadVariableOp2R
'block1a_se_expand/Conv2D/ReadVariableOp'block1a_se_expand/Conv2D/ReadVariableOp2T
(block1a_se_reduce/BiasAdd/ReadVariableOp(block1a_se_reduce/BiasAdd/ReadVariableOp2R
'block1a_se_reduce/Conv2D/ReadVariableOp'block1a_se_reduce/Conv2D/ReadVariableOp2X
*block1b_bn/FusedBatchNormV3/ReadVariableOp*block1b_bn/FusedBatchNormV3/ReadVariableOp2\
,block1b_bn/FusedBatchNormV3/ReadVariableOp_1,block1b_bn/FusedBatchNormV3/ReadVariableOp_126
block1b_bn/ReadVariableOpblock1b_bn/ReadVariableOp2:
block1b_bn/ReadVariableOp_1block1b_bn/ReadVariableOp_12R
'block1b_dwconv/depthwise/ReadVariableOp'block1b_dwconv/depthwise/ReadVariableOp2h
2block1b_project_bn/FusedBatchNormV3/ReadVariableOp2block1b_project_bn/FusedBatchNormV3/ReadVariableOp2l
4block1b_project_bn/FusedBatchNormV3/ReadVariableOp_14block1b_project_bn/FusedBatchNormV3/ReadVariableOp_12F
!block1b_project_bn/ReadVariableOp!block1b_project_bn/ReadVariableOp2J
#block1b_project_bn/ReadVariableOp_1#block1b_project_bn/ReadVariableOp_12X
*block1b_project_conv/Conv2D/ReadVariableOp*block1b_project_conv/Conv2D/ReadVariableOp2T
(block1b_se_expand/BiasAdd/ReadVariableOp(block1b_se_expand/BiasAdd/ReadVariableOp2R
'block1b_se_expand/Conv2D/ReadVariableOp'block1b_se_expand/Conv2D/ReadVariableOp2T
(block1b_se_reduce/BiasAdd/ReadVariableOp(block1b_se_reduce/BiasAdd/ReadVariableOp2R
'block1b_se_reduce/Conv2D/ReadVariableOp'block1b_se_reduce/Conv2D/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2L
$normalization/Reshape/ReadVariableOp$normalization/Reshape/ReadVariableOp2P
&normalization/Reshape_1/ReadVariableOp&normalization/Reshape_1/ReadVariableOp2@
output1/BiasAdd/ReadVariableOpoutput1/BiasAdd/ReadVariableOp2>
output1/MatMul/ReadVariableOpoutput1/MatMul/ReadVariableOp2@
output2/BiasAdd/ReadVariableOpoutput2/BiasAdd/ReadVariableOp2>
output2/MatMul/ReadVariableOpoutput2/MatMul/ReadVariableOp2@
output3/BiasAdd/ReadVariableOpoutput3/BiasAdd/ReadVariableOp2>
output3/MatMul/ReadVariableOpoutput3/MatMul/ReadVariableOp2@
output4/BiasAdd/ReadVariableOpoutput4/BiasAdd/ReadVariableOp2>
output4/MatMul/ReadVariableOpoutput4/MatMul/ReadVariableOp2R
'stem_bn/FusedBatchNormV3/ReadVariableOp'stem_bn/FusedBatchNormV3/ReadVariableOp2V
)stem_bn/FusedBatchNormV3/ReadVariableOp_1)stem_bn/FusedBatchNormV3/ReadVariableOp_120
stem_bn/ReadVariableOpstem_bn/ReadVariableOp24
stem_bn/ReadVariableOp_1stem_bn/ReadVariableOp_12B
stem_conv/Conv2D/ReadVariableOpstem_conv/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬
 
_user_specified_nameinputs
µ
O
3__inference_block1a_se_reshape_layer_call_fn_101531

inputs
identityÓ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_block1a_se_reshape_layer_call_and_return_conditional_losses_988332
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
û
{
5__inference_block1b_project_conv_layer_call_fn_101955

inputs
unknown
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_block1b_project_conv_layer_call_and_return_conditional_losses_992012
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ:22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
½
ð
M__inference_block1b_project_bn_layer_call_and_return_conditional_losses_98530

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3ì
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Û
B__inference_output3_layer_call_and_return_conditional_losses_99523

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ö
é
F__inference_block1a_bn_layer_call_and_return_conditional_losses_101471

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:(*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:(*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:(*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:(*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ì
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ(:(:(:(:(:*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ü
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ(::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs

¦
3__inference_block1a_project_bn_layer_call_fn_101651

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall±
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_block1a_project_bn_layer_call_and_return_conditional_losses_982992
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
½
ð
M__inference_block1a_project_bn_layer_call_and_return_conditional_losses_98299

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3ì
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Û
B__inference_output2_layer_call_and_return_conditional_losses_99549

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²
å
B__inference_stem_bn_layer_call_and_return_conditional_losses_98099

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:(*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:(*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:(*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:(*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(:(:(:(:(:*
epsilon%o:*
is_training( 2
FusedBatchNormV3ì
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs


2__inference_block1a_se_reduce_layer_call_fn_101556

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_block1a_se_reduce_layer_call_and_return_conditional_losses_988572
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ(::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
	
Û
B__inference_output4_layer_call_and_return_conditional_losses_99497

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø
¦
3__inference_block1b_project_bn_layer_call_fn_102079

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¡
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_block1b_project_bn_layer_call_and_return_conditional_losses_992482
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù

æ
M__inference_block1b_se_expand_layer_call_and_return_conditional_losses_101920

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddi
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoid
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¶
é
F__inference_block1b_bn_layer_call_and_return_conditional_losses_101806

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3ì
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Û
B__inference_output1_layer_call_and_return_conditional_losses_99575

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
é	
Û
B__inference_conv2d_layer_call_and_return_conditional_losses_102136

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
Ò
&__inference_model_layer_call_fn_101103

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51
identity

identity_1

identity_2

identity_3¢StatefulPartitionedCallí
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51*A
Tin:
826*
Tout
2*
_collective_manager_ids
 *`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*W
_read_only_resource_inputs9
75	
 !"#$%&'()*+,-./012345*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_999262
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*
_input_shapesô
ñ:ÿÿÿÿÿÿÿÿÿ¬¬:::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬
 
_user_specified_nameinputs


2__inference_block1a_se_expand_layer_call_fn_101576

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_block1a_se_expand_layer_call_and_return_conditional_losses_988842
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs


+__inference_block1b_bn_layer_call_fn_101850

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_block1b_bn_layer_call_and_return_conditional_losses_984482
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¶
é
F__inference_block1a_bn_layer_call_and_return_conditional_losses_101409

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:(*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:(*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:(*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:(*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(:(:(:(:(:*
epsilon%o:*
is_training( 2
FusedBatchNormV3ì
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
¶
é
F__inference_block1b_bn_layer_call_and_return_conditional_losses_101824

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3ì
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
§
I
-__inference_stem_conv_pad_layer_call_fn_98010

inputs
identityé
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_stem_conv_pad_layer_call_and_return_conditional_losses_980042
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


2__inference_block1b_se_expand_layer_call_fn_101929

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_block1b_se_expand_layer_call_and_return_conditional_losses_991632
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³
i
M__inference_block1a_se_squeeze_layer_call_and_return_conditional_losses_98235

inputs
identity
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³
æ
C__inference_stem_bn_layer_call_and_return_conditional_losses_101270

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:(*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:(*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:(*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:(*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(:(:(:(:(:*
epsilon%o:*
is_training( 2
FusedBatchNormV3ì
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
ü
N
2__inference_block1b_se_squeeze_layer_call_fn_98472

inputs
identityÔ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_block1b_se_squeeze_layer_call_and_return_conditional_losses_984662
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ
ª
O__inference_block1a_project_conv_layer_call_and_return_conditional_losses_98922

inputs"
conv2d_readvariableop_resource
identity¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:(*
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ(:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
ö
é
F__inference_block1b_bn_layer_call_and_return_conditional_losses_101762

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ì
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ü
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


2__inference_block1b_se_reduce_layer_call_fn_101909

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_block1b_se_reduce_layer_call_and_return_conditional_losses_991362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È

+__inference_block1b_bn_layer_call_fn_101788

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_block1b_bn_layer_call_and_return_conditional_losses_990432
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤à
¡
A__inference_model_layer_call_and_return_conditional_losses_100207

inputs1
-normalization_reshape_readvariableop_resource3
/normalization_reshape_1_readvariableop_resource
stem_conv_100064
stem_bn_100067
stem_bn_100069
stem_bn_100071
stem_bn_100073
block1a_dwconv_100077
block1a_bn_100080
block1a_bn_100082
block1a_bn_100084
block1a_bn_100086
block1a_se_reduce_100092
block1a_se_reduce_100094
block1a_se_expand_100097
block1a_se_expand_100099
block1a_project_conv_100103
block1a_project_bn_100106
block1a_project_bn_100108
block1a_project_bn_100110
block1a_project_bn_100112
block1b_dwconv_100115
block1b_bn_100118
block1b_bn_100120
block1b_bn_100122
block1b_bn_100124
block1b_se_reduce_100130
block1b_se_reduce_100132
block1b_se_expand_100135
block1b_se_expand_100137
block1b_project_conv_100141
block1b_project_bn_100144
block1b_project_bn_100146
block1b_project_bn_100148
block1b_project_bn_100150
conv2d_100156
conv2d_100158
dense_3_100163
dense_3_100165
dense_2_100168
dense_2_100170
dense_1_100173
dense_1_100175
dense_100178
dense_100180
output4_100183
output4_100185
output3_100188
output3_100190
output2_100193
output2_100195
output1_100198
output1_100200
identity

identity_1

identity_2

identity_3¢"block1a_bn/StatefulPartitionedCall¢&block1a_dwconv/StatefulPartitionedCall¢*block1a_project_bn/StatefulPartitionedCall¢,block1a_project_conv/StatefulPartitionedCall¢)block1a_se_expand/StatefulPartitionedCall¢)block1a_se_reduce/StatefulPartitionedCall¢"block1b_bn/StatefulPartitionedCall¢&block1b_dwconv/StatefulPartitionedCall¢*block1b_project_bn/StatefulPartitionedCall¢,block1b_project_conv/StatefulPartitionedCall¢)block1b_se_expand/StatefulPartitionedCall¢)block1b_se_reduce/StatefulPartitionedCall¢conv2d/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢$normalization/Reshape/ReadVariableOp¢&normalization/Reshape_1/ReadVariableOp¢output1/StatefulPartitionedCall¢output2/StatefulPartitionedCall¢output3/StatefulPartitionedCall¢output4/StatefulPartitionedCall¢stem_bn/StatefulPartitionedCall¢!stem_conv/StatefulPartitionedCalli
rescaling/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;2
rescaling/Cast/xm
rescaling/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rescaling/Cast_1/x
rescaling/mulMulinputsrescaling/Cast/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬2
rescaling/mul
rescaling/addAddV2rescaling/mul:z:0rescaling/Cast_1/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬2
rescaling/add¶
$normalization/Reshape/ReadVariableOpReadVariableOp-normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02&
$normalization/Reshape/ReadVariableOp
normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
normalization/Reshape/shape¾
normalization/ReshapeReshape,normalization/Reshape/ReadVariableOp:value:0$normalization/Reshape/shape:output:0*
T0*&
_output_shapes
:2
normalization/Reshape¼
&normalization/Reshape_1/ReadVariableOpReadVariableOp/normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization/Reshape_1/ReadVariableOp
normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
normalization/Reshape_1/shapeÆ
normalization/Reshape_1Reshape.normalization/Reshape_1/ReadVariableOp:value:0&normalization/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
normalization/Reshape_1
normalization/subSubrescaling/add:z:0normalization/Reshape:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬2
normalization/sub
normalization/SqrtSqrt normalization/Reshape_1:output:0*
T0*&
_output_shapes
:2
normalization/Sqrtw
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö32
normalization/Maximum/y¤
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*&
_output_shapes
:2
normalization/Maximum§
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬2
normalization/truedivÿ
stem_conv_pad/PartitionedCallPartitionedCallnormalization/truediv:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­­* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_stem_conv_pad_layer_call_and_return_conditional_losses_980042
stem_conv_pad/PartitionedCall®
!stem_conv/StatefulPartitionedCallStatefulPartitionedCall&stem_conv_pad/PartitionedCall:output:0stem_conv_100064*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_stem_conv_layer_call_and_return_conditional_losses_986252#
!stem_conv/StatefulPartitionedCallà
stem_bn/StatefulPartitionedCallStatefulPartitionedCall*stem_conv/StatefulPartitionedCall:output:0stem_bn_100067stem_bn_100069stem_bn_100071stem_bn_100073*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_stem_bn_layer_call_and_return_conditional_losses_986722!
stem_bn/StatefulPartitionedCall
stem_activation/PartitionedCallPartitionedCall(stem_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_stem_activation_layer_call_and_return_conditional_losses_987182!
stem_activation/PartitionedCallÄ
&block1a_dwconv/StatefulPartitionedCallStatefulPartitionedCall(stem_activation/PartitionedCall:output:0block1a_dwconv_100077*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_block1a_dwconv_layer_call_and_return_conditional_losses_981202(
&block1a_dwconv/StatefulPartitionedCallú
"block1a_bn/StatefulPartitionedCallStatefulPartitionedCall/block1a_dwconv/StatefulPartitionedCall:output:0block1a_bn_100080block1a_bn_100082block1a_bn_100084block1a_bn_100086*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_block1a_bn_layer_call_and_return_conditional_losses_987642$
"block1a_bn/StatefulPartitionedCall 
"block1a_activation/PartitionedCallPartitionedCall+block1a_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_block1a_activation_layer_call_and_return_conditional_losses_988102$
"block1a_activation/PartitionedCall
"block1a_se_squeeze/PartitionedCallPartitionedCall+block1a_activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_block1a_se_squeeze_layer_call_and_return_conditional_losses_982352$
"block1a_se_squeeze/PartitionedCall
"block1a_se_reshape/PartitionedCallPartitionedCall+block1a_se_squeeze/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_block1a_se_reshape_layer_call_and_return_conditional_losses_988332$
"block1a_se_reshape/PartitionedCallí
)block1a_se_reduce/StatefulPartitionedCallStatefulPartitionedCall+block1a_se_reshape/PartitionedCall:output:0block1a_se_reduce_100092block1a_se_reduce_100094*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_block1a_se_reduce_layer_call_and_return_conditional_losses_988572+
)block1a_se_reduce/StatefulPartitionedCallô
)block1a_se_expand/StatefulPartitionedCallStatefulPartitionedCall2block1a_se_reduce/StatefulPartitionedCall:output:0block1a_se_expand_100097block1a_se_expand_100099*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_block1a_se_expand_layer_call_and_return_conditional_losses_988842+
)block1a_se_expand/StatefulPartitionedCallÒ
!block1a_se_excite/PartitionedCallPartitionedCall+block1a_activation/PartitionedCall:output:02block1a_se_expand/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_block1a_se_excite_layer_call_and_return_conditional_losses_989062#
!block1a_se_excite/PartitionedCallÞ
,block1a_project_conv/StatefulPartitionedCallStatefulPartitionedCall*block1a_se_excite/PartitionedCall:output:0block1a_project_conv_100103*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_block1a_project_conv_layer_call_and_return_conditional_losses_989222.
,block1a_project_conv/StatefulPartitionedCall¸
*block1a_project_bn/StatefulPartitionedCallStatefulPartitionedCall5block1a_project_conv/StatefulPartitionedCall:output:0block1a_project_bn_100106block1a_project_bn_100108block1a_project_bn_100110block1a_project_bn_100112*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_block1a_project_bn_layer_call_and_return_conditional_losses_989692,
*block1a_project_bn/StatefulPartitionedCallÏ
&block1b_dwconv/StatefulPartitionedCallStatefulPartitionedCall3block1a_project_bn/StatefulPartitionedCall:output:0block1b_dwconv_100115*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_block1b_dwconv_layer_call_and_return_conditional_losses_983512(
&block1b_dwconv/StatefulPartitionedCallú
"block1b_bn/StatefulPartitionedCallStatefulPartitionedCall/block1b_dwconv/StatefulPartitionedCall:output:0block1b_bn_100118block1b_bn_100120block1b_bn_100122block1b_bn_100124*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_block1b_bn_layer_call_and_return_conditional_losses_990432$
"block1b_bn/StatefulPartitionedCall 
"block1b_activation/PartitionedCallPartitionedCall+block1b_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_block1b_activation_layer_call_and_return_conditional_losses_990892$
"block1b_activation/PartitionedCall
"block1b_se_squeeze/PartitionedCallPartitionedCall+block1b_activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_block1b_se_squeeze_layer_call_and_return_conditional_losses_984662$
"block1b_se_squeeze/PartitionedCall
"block1b_se_reshape/PartitionedCallPartitionedCall+block1b_se_squeeze/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_block1b_se_reshape_layer_call_and_return_conditional_losses_991122$
"block1b_se_reshape/PartitionedCallí
)block1b_se_reduce/StatefulPartitionedCallStatefulPartitionedCall+block1b_se_reshape/PartitionedCall:output:0block1b_se_reduce_100130block1b_se_reduce_100132*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_block1b_se_reduce_layer_call_and_return_conditional_losses_991362+
)block1b_se_reduce/StatefulPartitionedCallô
)block1b_se_expand/StatefulPartitionedCallStatefulPartitionedCall2block1b_se_reduce/StatefulPartitionedCall:output:0block1b_se_expand_100135block1b_se_expand_100137*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_block1b_se_expand_layer_call_and_return_conditional_losses_991632+
)block1b_se_expand/StatefulPartitionedCallÒ
!block1b_se_excite/PartitionedCallPartitionedCall+block1b_activation/PartitionedCall:output:02block1b_se_expand/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_block1b_se_excite_layer_call_and_return_conditional_losses_991852#
!block1b_se_excite/PartitionedCallÞ
,block1b_project_conv/StatefulPartitionedCallStatefulPartitionedCall*block1b_se_excite/PartitionedCall:output:0block1b_project_conv_100141*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_block1b_project_conv_layer_call_and_return_conditional_losses_992012.
,block1b_project_conv/StatefulPartitionedCall¸
*block1b_project_bn/StatefulPartitionedCallStatefulPartitionedCall5block1b_project_conv/StatefulPartitionedCall:output:0block1b_project_bn_100144block1b_project_bn_100146block1b_project_bn_100148block1b_project_bn_100150*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_block1b_project_bn_layer_call_and_return_conditional_losses_992482,
*block1b_project_bn/StatefulPartitionedCall
block1b_drop/PartitionedCallPartitionedCall3block1b_project_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_block1b_drop_layer_call_and_return_conditional_losses_993092
block1b_drop/PartitionedCall»
block1b_add/PartitionedCallPartitionedCall%block1b_drop/PartitionedCall:output:03block1a_project_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block1b_add_layer_call_and_return_conditional_losses_993282
block1b_add/PartitionedCall
!average_pooling2d/PartitionedCallPartitionedCall$block1b_add/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_985782#
!average_pooling2d/PartitionedCallµ
conv2d/StatefulPartitionedCallStatefulPartitionedCall*average_pooling2d/PartitionedCall:output:0conv2d_100156conv2d_100158*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_993482 
conv2d/StatefulPartitionedCall
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_985902
max_pooling2d/PartitionedCallñ
flatten/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_993712
flatten/PartitionedCall¨
dense_3/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_3_100163dense_3_100165*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_993902!
dense_3/StatefulPartitionedCall¨
dense_2/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_2_100168dense_2_100170*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_994172!
dense_2/StatefulPartitionedCall¨
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_100173dense_1_100175*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_994442!
dense_1/StatefulPartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_100178dense_100180*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_994712
dense/StatefulPartitionedCall°
output4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0output4_100183output4_100185*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_output4_layer_call_and_return_conditional_losses_994972!
output4/StatefulPartitionedCall°
output3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0output3_100188output3_100190*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_output3_layer_call_and_return_conditional_losses_995232!
output3/StatefulPartitionedCall°
output2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0output2_100193output2_100195*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_output2_layer_call_and_return_conditional_losses_995492!
output2/StatefulPartitionedCall®
output1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0output1_100198output1_100200*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_output1_layer_call_and_return_conditional_losses_995752!
output1/StatefulPartitionedCallÅ
IdentityIdentity(output1/StatefulPartitionedCall:output:0#^block1a_bn/StatefulPartitionedCall'^block1a_dwconv/StatefulPartitionedCall+^block1a_project_bn/StatefulPartitionedCall-^block1a_project_conv/StatefulPartitionedCall*^block1a_se_expand/StatefulPartitionedCall*^block1a_se_reduce/StatefulPartitionedCall#^block1b_bn/StatefulPartitionedCall'^block1b_dwconv/StatefulPartitionedCall+^block1b_project_bn/StatefulPartitionedCall-^block1b_project_conv/StatefulPartitionedCall*^block1b_se_expand/StatefulPartitionedCall*^block1b_se_reduce/StatefulPartitionedCall^conv2d/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp ^output1/StatefulPartitionedCall ^output2/StatefulPartitionedCall ^output3/StatefulPartitionedCall ^output4/StatefulPartitionedCall ^stem_bn/StatefulPartitionedCall"^stem_conv/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÉ

Identity_1Identity(output2/StatefulPartitionedCall:output:0#^block1a_bn/StatefulPartitionedCall'^block1a_dwconv/StatefulPartitionedCall+^block1a_project_bn/StatefulPartitionedCall-^block1a_project_conv/StatefulPartitionedCall*^block1a_se_expand/StatefulPartitionedCall*^block1a_se_reduce/StatefulPartitionedCall#^block1b_bn/StatefulPartitionedCall'^block1b_dwconv/StatefulPartitionedCall+^block1b_project_bn/StatefulPartitionedCall-^block1b_project_conv/StatefulPartitionedCall*^block1b_se_expand/StatefulPartitionedCall*^block1b_se_reduce/StatefulPartitionedCall^conv2d/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp ^output1/StatefulPartitionedCall ^output2/StatefulPartitionedCall ^output3/StatefulPartitionedCall ^output4/StatefulPartitionedCall ^stem_bn/StatefulPartitionedCall"^stem_conv/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1É

Identity_2Identity(output3/StatefulPartitionedCall:output:0#^block1a_bn/StatefulPartitionedCall'^block1a_dwconv/StatefulPartitionedCall+^block1a_project_bn/StatefulPartitionedCall-^block1a_project_conv/StatefulPartitionedCall*^block1a_se_expand/StatefulPartitionedCall*^block1a_se_reduce/StatefulPartitionedCall#^block1b_bn/StatefulPartitionedCall'^block1b_dwconv/StatefulPartitionedCall+^block1b_project_bn/StatefulPartitionedCall-^block1b_project_conv/StatefulPartitionedCall*^block1b_se_expand/StatefulPartitionedCall*^block1b_se_reduce/StatefulPartitionedCall^conv2d/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp ^output1/StatefulPartitionedCall ^output2/StatefulPartitionedCall ^output3/StatefulPartitionedCall ^output4/StatefulPartitionedCall ^stem_bn/StatefulPartitionedCall"^stem_conv/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2É

Identity_3Identity(output4/StatefulPartitionedCall:output:0#^block1a_bn/StatefulPartitionedCall'^block1a_dwconv/StatefulPartitionedCall+^block1a_project_bn/StatefulPartitionedCall-^block1a_project_conv/StatefulPartitionedCall*^block1a_se_expand/StatefulPartitionedCall*^block1a_se_reduce/StatefulPartitionedCall#^block1b_bn/StatefulPartitionedCall'^block1b_dwconv/StatefulPartitionedCall+^block1b_project_bn/StatefulPartitionedCall-^block1b_project_conv/StatefulPartitionedCall*^block1b_se_expand/StatefulPartitionedCall*^block1b_se_reduce/StatefulPartitionedCall^conv2d/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp ^output1/StatefulPartitionedCall ^output2/StatefulPartitionedCall ^output3/StatefulPartitionedCall ^output4/StatefulPartitionedCall ^stem_bn/StatefulPartitionedCall"^stem_conv/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*
_input_shapesô
ñ:ÿÿÿÿÿÿÿÿÿ¬¬:::::::::::::::::::::::::::::::::::::::::::::::::::::2H
"block1a_bn/StatefulPartitionedCall"block1a_bn/StatefulPartitionedCall2P
&block1a_dwconv/StatefulPartitionedCall&block1a_dwconv/StatefulPartitionedCall2X
*block1a_project_bn/StatefulPartitionedCall*block1a_project_bn/StatefulPartitionedCall2\
,block1a_project_conv/StatefulPartitionedCall,block1a_project_conv/StatefulPartitionedCall2V
)block1a_se_expand/StatefulPartitionedCall)block1a_se_expand/StatefulPartitionedCall2V
)block1a_se_reduce/StatefulPartitionedCall)block1a_se_reduce/StatefulPartitionedCall2H
"block1b_bn/StatefulPartitionedCall"block1b_bn/StatefulPartitionedCall2P
&block1b_dwconv/StatefulPartitionedCall&block1b_dwconv/StatefulPartitionedCall2X
*block1b_project_bn/StatefulPartitionedCall*block1b_project_bn/StatefulPartitionedCall2\
,block1b_project_conv/StatefulPartitionedCall,block1b_project_conv/StatefulPartitionedCall2V
)block1b_se_expand/StatefulPartitionedCall)block1b_se_expand/StatefulPartitionedCall2V
)block1b_se_reduce/StatefulPartitionedCall)block1b_se_reduce/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2L
$normalization/Reshape/ReadVariableOp$normalization/Reshape/ReadVariableOp2P
&normalization/Reshape_1/ReadVariableOp&normalization/Reshape_1/ReadVariableOp2B
output1/StatefulPartitionedCalloutput1/StatefulPartitionedCall2B
output2/StatefulPartitionedCalloutput2/StatefulPartitionedCall2B
output3/StatefulPartitionedCalloutput3/StatefulPartitionedCall2B
output4/StatefulPartitionedCalloutput4/StatefulPartitionedCall2B
stem_bn/StatefulPartitionedCallstem_bn/StatefulPartitionedCall2F
!stem_conv/StatefulPartitionedCall!stem_conv/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬
 
_user_specified_nameinputs
ï
j
N__inference_block1a_se_reshape_layer_call_and_return_conditional_losses_101526

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :(2
Reshape/shape/3º
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
ò
å
B__inference_stem_bn_layer_call_and_return_conditional_losses_98654

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:(*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:(*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:(*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:(*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ì
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ(:(:(:(:(:*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ü
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ(::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
Ø
¦
3__inference_block1a_project_bn_layer_call_fn_101713

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¡
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_block1a_project_bn_layer_call_and_return_conditional_losses_989512
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í	
Ù
@__inference_dense_layer_call_and_return_conditional_losses_99471

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Å	
ª
I__inference_block1a_dwconv_layer_call_and_return_conditional_losses_98120

inputs%
!depthwise_readvariableop_resource
identity¢depthwise/ReadVariableOp
depthwise/ReadVariableOpReadVariableOp!depthwise_readvariableop_resource*&
_output_shapes
:(*
dtype02
depthwise/ReadVariableOp{
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      (      2
depthwise/Shape
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
depthwise/dilation_rateÍ
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(*
paddingSAME*
strides
2
	depthwise
IdentityIdentitydepthwise:output:0^depthwise/ReadVariableOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(:24
depthwise/ReadVariableOpdepthwise/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
Ù

æ
M__inference_block1a_se_expand_layer_call_and_return_conditional_losses_101567

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
(*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2	
BiasAddi
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2	
Sigmoid
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Û
}
(__inference_dense_1_layer_call_fn_102196

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_994442
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
õ
è
E__inference_block1b_bn_layer_call_and_return_conditional_losses_99043

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ì
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ü
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì
 
E__inference_stem_conv_layer_call_and_return_conditional_losses_101227

inputs"
conv2d_readvariableop_resource
identity¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:(*
dtype02
Conv2D/ReadVariableOp¦
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
paddingVALID*
strides
2
Conv2D
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ­­:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­­
 
_user_specified_nameinputs
ý
ð
M__inference_block1b_project_bn_layer_call_and_return_conditional_losses_99230

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ì
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ü
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø
¦
3__inference_block1b_project_bn_layer_call_fn_102066

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¡
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_block1b_project_bn_layer_call_and_return_conditional_losses_992302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù
}
(__inference_output1_layer_call_fn_102255

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_output1_layer_call_and_return_conditional_losses_995752
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö
«
P__inference_block1a_project_conv_layer_call_and_return_conditional_losses_101595

inputs"
conv2d_readvariableop_resource
identity¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:(*
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ(:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
Ù
}
(__inference_output3_layer_call_fn_102293

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_output3_layer_call_and_return_conditional_losses_995232
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
þ
ñ
N__inference_block1a_project_bn_layer_call_and_return_conditional_losses_101700

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ì
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ü
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


+__inference_block1b_bn_layer_call_fn_101837

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_block1b_bn_layer_call_and_return_conditional_losses_984172
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

l
N__inference_block1a_activation_layer_call_and_return_conditional_losses_101507

inputs

identity_1a
SigmoidSigmoidinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2	
Sigmoidb
mulMulinputsSigmoid:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
mule
IdentityIdentitymul:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity¿
	IdentityN	IdentityNmul:z:0inputs*
T
2*,
_gradient_op_typeCustomGradient-101500*N
_output_shapes<
::ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(2
	IdentityNt

Identity_1IdentityIdentityN:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity_1"!

identity_1Identity_1:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs

k
M__inference_block1b_activation_layer_call_and_return_conditional_losses_99089

inputs

identity_1a
SigmoidSigmoidinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidb
mulMulinputsSigmoid:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mule
IdentityIdentitymul:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¾
	IdentityN	IdentityNmul:z:0inputs*
T
2*+
_gradient_op_typeCustomGradient-99082*N
_output_shapes<
::ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ2
	IdentityNt

Identity_1IdentityIdentityN:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


+__inference_block1a_bn_layer_call_fn_101435

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_block1a_bn_layer_call_and_return_conditional_losses_982172
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
¡
D
(__inference_flatten_layer_call_fn_102156

inputs
identityÁ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_993712
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Û
}
(__inference_dense_3_layer_call_fn_102236

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_993902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¿
Ð*
__inference__traced_save_102653
file_prefix1
-savev2_normalization_mean_read_readvariableop5
1savev2_normalization_variance_read_readvariableop2
.savev2_normalization_count_read_readvariableop	/
+savev2_stem_conv_kernel_read_readvariableop,
(savev2_stem_bn_gamma_read_readvariableop+
'savev2_stem_bn_beta_read_readvariableop2
.savev2_stem_bn_moving_mean_read_readvariableop6
2savev2_stem_bn_moving_variance_read_readvariableop>
:savev2_block1a_dwconv_depthwise_kernel_read_readvariableop/
+savev2_block1a_bn_gamma_read_readvariableop.
*savev2_block1a_bn_beta_read_readvariableop5
1savev2_block1a_bn_moving_mean_read_readvariableop9
5savev2_block1a_bn_moving_variance_read_readvariableop7
3savev2_block1a_se_reduce_kernel_read_readvariableop5
1savev2_block1a_se_reduce_bias_read_readvariableop7
3savev2_block1a_se_expand_kernel_read_readvariableop5
1savev2_block1a_se_expand_bias_read_readvariableop:
6savev2_block1a_project_conv_kernel_read_readvariableop7
3savev2_block1a_project_bn_gamma_read_readvariableop6
2savev2_block1a_project_bn_beta_read_readvariableop=
9savev2_block1a_project_bn_moving_mean_read_readvariableopA
=savev2_block1a_project_bn_moving_variance_read_readvariableop>
:savev2_block1b_dwconv_depthwise_kernel_read_readvariableop/
+savev2_block1b_bn_gamma_read_readvariableop.
*savev2_block1b_bn_beta_read_readvariableop5
1savev2_block1b_bn_moving_mean_read_readvariableop9
5savev2_block1b_bn_moving_variance_read_readvariableop7
3savev2_block1b_se_reduce_kernel_read_readvariableop5
1savev2_block1b_se_reduce_bias_read_readvariableop7
3savev2_block1b_se_expand_kernel_read_readvariableop5
1savev2_block1b_se_expand_bias_read_readvariableop:
6savev2_block1b_project_conv_kernel_read_readvariableop7
3savev2_block1b_project_bn_gamma_read_readvariableop6
2savev2_block1b_project_bn_beta_read_readvariableop=
9savev2_block1b_project_bn_moving_mean_read_readvariableopA
=savev2_block1b_project_bn_moving_variance_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop-
)savev2_output1_kernel_read_readvariableop+
'savev2_output1_bias_read_readvariableop-
)savev2_output2_kernel_read_readvariableop+
'savev2_output2_bias_read_readvariableop-
)savev2_output3_kernel_read_readvariableop+
'savev2_output3_bias_read_readvariableop-
)savev2_output4_kernel_read_readvariableop+
'savev2_output4_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_3_read_readvariableop&
"savev2_count_3_read_readvariableop&
"savev2_total_4_read_readvariableop&
"savev2_count_4_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop4
0savev2_adam_dense_3_kernel_m_read_readvariableop2
.savev2_adam_dense_3_bias_m_read_readvariableop4
0savev2_adam_output1_kernel_m_read_readvariableop2
.savev2_adam_output1_bias_m_read_readvariableop4
0savev2_adam_output2_kernel_m_read_readvariableop2
.savev2_adam_output2_bias_m_read_readvariableop4
0savev2_adam_output3_kernel_m_read_readvariableop2
.savev2_adam_output3_bias_m_read_readvariableop4
0savev2_adam_output4_kernel_m_read_readvariableop2
.savev2_adam_output4_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop4
0savev2_adam_dense_3_kernel_v_read_readvariableop2
.savev2_adam_dense_3_bias_v_read_readvariableop4
0savev2_adam_output1_kernel_v_read_readvariableop2
.savev2_adam_output1_bias_v_read_readvariableop4
0savev2_adam_output2_kernel_v_read_readvariableop2
.savev2_adam_output2_bias_v_read_readvariableop4
0savev2_adam_output3_kernel_v_read_readvariableop2
.savev2_adam_output3_bias_v_read_readvariableop4
0savev2_adam_output4_kernel_v_read_readvariableop2
.savev2_adam_output4_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename7
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:j*
dtype0*£6
value6B6jB4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-3/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-9/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesß
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:j*
dtype0*é
valueßBÜjB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesã(
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0-savev2_normalization_mean_read_readvariableop1savev2_normalization_variance_read_readvariableop.savev2_normalization_count_read_readvariableop+savev2_stem_conv_kernel_read_readvariableop(savev2_stem_bn_gamma_read_readvariableop'savev2_stem_bn_beta_read_readvariableop.savev2_stem_bn_moving_mean_read_readvariableop2savev2_stem_bn_moving_variance_read_readvariableop:savev2_block1a_dwconv_depthwise_kernel_read_readvariableop+savev2_block1a_bn_gamma_read_readvariableop*savev2_block1a_bn_beta_read_readvariableop1savev2_block1a_bn_moving_mean_read_readvariableop5savev2_block1a_bn_moving_variance_read_readvariableop3savev2_block1a_se_reduce_kernel_read_readvariableop1savev2_block1a_se_reduce_bias_read_readvariableop3savev2_block1a_se_expand_kernel_read_readvariableop1savev2_block1a_se_expand_bias_read_readvariableop6savev2_block1a_project_conv_kernel_read_readvariableop3savev2_block1a_project_bn_gamma_read_readvariableop2savev2_block1a_project_bn_beta_read_readvariableop9savev2_block1a_project_bn_moving_mean_read_readvariableop=savev2_block1a_project_bn_moving_variance_read_readvariableop:savev2_block1b_dwconv_depthwise_kernel_read_readvariableop+savev2_block1b_bn_gamma_read_readvariableop*savev2_block1b_bn_beta_read_readvariableop1savev2_block1b_bn_moving_mean_read_readvariableop5savev2_block1b_bn_moving_variance_read_readvariableop3savev2_block1b_se_reduce_kernel_read_readvariableop1savev2_block1b_se_reduce_bias_read_readvariableop3savev2_block1b_se_expand_kernel_read_readvariableop1savev2_block1b_se_expand_bias_read_readvariableop6savev2_block1b_project_conv_kernel_read_readvariableop3savev2_block1b_project_bn_gamma_read_readvariableop2savev2_block1b_project_bn_beta_read_readvariableop9savev2_block1b_project_bn_moving_mean_read_readvariableop=savev2_block1b_project_bn_moving_variance_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop)savev2_output1_kernel_read_readvariableop'savev2_output1_bias_read_readvariableop)savev2_output2_kernel_read_readvariableop'savev2_output2_bias_read_readvariableop)savev2_output3_kernel_read_readvariableop'savev2_output3_bias_read_readvariableop)savev2_output4_kernel_read_readvariableop'savev2_output4_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_3_read_readvariableop"savev2_count_3_read_readvariableop"savev2_total_4_read_readvariableop"savev2_count_4_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop0savev2_adam_dense_3_kernel_m_read_readvariableop.savev2_adam_dense_3_bias_m_read_readvariableop0savev2_adam_output1_kernel_m_read_readvariableop.savev2_adam_output1_bias_m_read_readvariableop0savev2_adam_output2_kernel_m_read_readvariableop.savev2_adam_output2_bias_m_read_readvariableop0savev2_adam_output3_kernel_m_read_readvariableop.savev2_adam_output3_bias_m_read_readvariableop0savev2_adam_output4_kernel_m_read_readvariableop.savev2_adam_output4_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableop0savev2_adam_dense_3_kernel_v_read_readvariableop.savev2_adam_dense_3_bias_v_read_readvariableop0savev2_adam_output1_kernel_v_read_readvariableop.savev2_adam_output1_bias_v_read_readvariableop0savev2_adam_output2_kernel_v_read_readvariableop.savev2_adam_output2_bias_v_read_readvariableop0savev2_adam_output3_kernel_v_read_readvariableop.savev2_adam_output3_bias_v_read_readvariableop0savev2_adam_output4_kernel_v_read_readvariableop.savev2_adam_output4_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *x
dtypesn
l2j		2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*Ë
_input_shapes¹
¶: ::: :(:(:(:(:(:(:(:(:(:(:(
:
:
(:(:(:::::::::::::::::::::	::	::	::	:::::::::: : : : : : : : : : : : : : : :::	::	::	::	::::::::::::	::	::	::	:::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :,(
&
_output_shapes
:(: 

_output_shapes
:(: 

_output_shapes
:(: 

_output_shapes
:(: 

_output_shapes
:(:,	(
&
_output_shapes
:(: 


_output_shapes
:(: 

_output_shapes
:(: 

_output_shapes
:(: 

_output_shapes
:(:,(
&
_output_shapes
:(
: 

_output_shapes
:
:,(
&
_output_shapes
:
(: 

_output_shapes
:(:,(
&
_output_shapes
:(: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::, (
&
_output_shapes
:: !

_output_shapes
:: "

_output_shapes
:: #

_output_shapes
:: $

_output_shapes
::,%(
&
_output_shapes
:: &

_output_shapes
::%'!

_output_shapes
:	: (

_output_shapes
::%)!

_output_shapes
:	: *

_output_shapes
::%+!

_output_shapes
:	: ,

_output_shapes
::%-!

_output_shapes
:	: .

_output_shapes
::$/ 

_output_shapes

:: 0

_output_shapes
::$1 

_output_shapes

:: 2

_output_shapes
::$3 

_output_shapes

:: 4

_output_shapes
::$5 

_output_shapes

:: 6

_output_shapes
::7

_output_shapes
: :8

_output_shapes
: :9

_output_shapes
: ::

_output_shapes
: :;

_output_shapes
: :<

_output_shapes
: :=

_output_shapes
: :>

_output_shapes
: :?

_output_shapes
: :@

_output_shapes
: :A

_output_shapes
: :B

_output_shapes
: :C

_output_shapes
: :D

_output_shapes
: :E

_output_shapes
: :,F(
&
_output_shapes
:: G

_output_shapes
::%H!

_output_shapes
:	: I

_output_shapes
::%J!

_output_shapes
:	: K

_output_shapes
::%L!

_output_shapes
:	: M

_output_shapes
::%N!

_output_shapes
:	: O

_output_shapes
::$P 

_output_shapes

:: Q

_output_shapes
::$R 

_output_shapes

:: S

_output_shapes
::$T 

_output_shapes

:: U

_output_shapes
::$V 

_output_shapes

:: W

_output_shapes
::,X(
&
_output_shapes
:: Y

_output_shapes
::%Z!

_output_shapes
:	: [

_output_shapes
::%\!

_output_shapes
:	: ]

_output_shapes
::%^!

_output_shapes
:	: _

_output_shapes
::%`!

_output_shapes
:	: a

_output_shapes
::$b 

_output_shapes

:: c

_output_shapes
::$d 

_output_shapes

:: e

_output_shapes
::$f 

_output_shapes

:: g

_output_shapes
::$h 

_output_shapes

:: i

_output_shapes
::j

_output_shapes
: 
þ
ñ
N__inference_block1b_project_bn_layer_call_and_return_conditional_losses_102053

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ì
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ü
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ý
ð
M__inference_block1b_project_bn_layer_call_and_return_conditional_losses_99248

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ì
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ü
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
O
3__inference_block1b_se_reshape_layer_call_fn_101884

inputs
identityÓ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_block1b_se_reshape_layer_call_and_return_conditional_losses_991122
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

è
M__inference_block1a_se_reduce_layer_call_and_return_conditional_losses_101547

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource

identity_1¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:(
*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
BiasAddi
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
Sigmoidj
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
mulc
IdentityIdentitymul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

IdentityÅ
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-101540*J
_output_shapes8
6:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
2
	IdentityN£

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity_1"!

identity_1Identity_1:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ(::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
³
æ
C__inference_stem_bn_layer_call_and_return_conditional_losses_101252

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:(*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:(*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:(*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:(*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(:(:(:(:(:*
epsilon%o:*
is_training( 2
FusedBatchNormV3ì
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs

h
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_98578

inputs
identity¶
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
AvgPool
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó
æ
C__inference_stem_bn_layer_call_and_return_conditional_losses_101332

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:(*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:(*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:(*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:(*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ì
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ(:(:(:(:(:*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ü
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ(::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
±
Ç6
"__inference__traced_restore_102978
file_prefix'
#assignvariableop_normalization_mean-
)assignvariableop_1_normalization_variance*
&assignvariableop_2_normalization_count'
#assignvariableop_3_stem_conv_kernel$
 assignvariableop_4_stem_bn_gamma#
assignvariableop_5_stem_bn_beta*
&assignvariableop_6_stem_bn_moving_mean.
*assignvariableop_7_stem_bn_moving_variance6
2assignvariableop_8_block1a_dwconv_depthwise_kernel'
#assignvariableop_9_block1a_bn_gamma'
#assignvariableop_10_block1a_bn_beta.
*assignvariableop_11_block1a_bn_moving_mean2
.assignvariableop_12_block1a_bn_moving_variance0
,assignvariableop_13_block1a_se_reduce_kernel.
*assignvariableop_14_block1a_se_reduce_bias0
,assignvariableop_15_block1a_se_expand_kernel.
*assignvariableop_16_block1a_se_expand_bias3
/assignvariableop_17_block1a_project_conv_kernel0
,assignvariableop_18_block1a_project_bn_gamma/
+assignvariableop_19_block1a_project_bn_beta6
2assignvariableop_20_block1a_project_bn_moving_mean:
6assignvariableop_21_block1a_project_bn_moving_variance7
3assignvariableop_22_block1b_dwconv_depthwise_kernel(
$assignvariableop_23_block1b_bn_gamma'
#assignvariableop_24_block1b_bn_beta.
*assignvariableop_25_block1b_bn_moving_mean2
.assignvariableop_26_block1b_bn_moving_variance0
,assignvariableop_27_block1b_se_reduce_kernel.
*assignvariableop_28_block1b_se_reduce_bias0
,assignvariableop_29_block1b_se_expand_kernel.
*assignvariableop_30_block1b_se_expand_bias3
/assignvariableop_31_block1b_project_conv_kernel0
,assignvariableop_32_block1b_project_bn_gamma/
+assignvariableop_33_block1b_project_bn_beta6
2assignvariableop_34_block1b_project_bn_moving_mean:
6assignvariableop_35_block1b_project_bn_moving_variance%
!assignvariableop_36_conv2d_kernel#
assignvariableop_37_conv2d_bias$
 assignvariableop_38_dense_kernel"
assignvariableop_39_dense_bias&
"assignvariableop_40_dense_1_kernel$
 assignvariableop_41_dense_1_bias&
"assignvariableop_42_dense_2_kernel$
 assignvariableop_43_dense_2_bias&
"assignvariableop_44_dense_3_kernel$
 assignvariableop_45_dense_3_bias&
"assignvariableop_46_output1_kernel$
 assignvariableop_47_output1_bias&
"assignvariableop_48_output2_kernel$
 assignvariableop_49_output2_bias&
"assignvariableop_50_output3_kernel$
 assignvariableop_51_output3_bias&
"assignvariableop_52_output4_kernel$
 assignvariableop_53_output4_bias!
assignvariableop_54_adam_iter#
assignvariableop_55_adam_beta_1#
assignvariableop_56_adam_beta_2"
assignvariableop_57_adam_decay*
&assignvariableop_58_adam_learning_rate
assignvariableop_59_total
assignvariableop_60_count
assignvariableop_61_total_1
assignvariableop_62_count_1
assignvariableop_63_total_2
assignvariableop_64_count_2
assignvariableop_65_total_3
assignvariableop_66_count_3
assignvariableop_67_total_4
assignvariableop_68_count_4,
(assignvariableop_69_adam_conv2d_kernel_m*
&assignvariableop_70_adam_conv2d_bias_m+
'assignvariableop_71_adam_dense_kernel_m)
%assignvariableop_72_adam_dense_bias_m-
)assignvariableop_73_adam_dense_1_kernel_m+
'assignvariableop_74_adam_dense_1_bias_m-
)assignvariableop_75_adam_dense_2_kernel_m+
'assignvariableop_76_adam_dense_2_bias_m-
)assignvariableop_77_adam_dense_3_kernel_m+
'assignvariableop_78_adam_dense_3_bias_m-
)assignvariableop_79_adam_output1_kernel_m+
'assignvariableop_80_adam_output1_bias_m-
)assignvariableop_81_adam_output2_kernel_m+
'assignvariableop_82_adam_output2_bias_m-
)assignvariableop_83_adam_output3_kernel_m+
'assignvariableop_84_adam_output3_bias_m-
)assignvariableop_85_adam_output4_kernel_m+
'assignvariableop_86_adam_output4_bias_m,
(assignvariableop_87_adam_conv2d_kernel_v*
&assignvariableop_88_adam_conv2d_bias_v+
'assignvariableop_89_adam_dense_kernel_v)
%assignvariableop_90_adam_dense_bias_v-
)assignvariableop_91_adam_dense_1_kernel_v+
'assignvariableop_92_adam_dense_1_bias_v-
)assignvariableop_93_adam_dense_2_kernel_v+
'assignvariableop_94_adam_dense_2_bias_v-
)assignvariableop_95_adam_dense_3_kernel_v+
'assignvariableop_96_adam_dense_3_bias_v-
)assignvariableop_97_adam_output1_kernel_v+
'assignvariableop_98_adam_output1_bias_v-
)assignvariableop_99_adam_output2_kernel_v,
(assignvariableop_100_adam_output2_bias_v.
*assignvariableop_101_adam_output3_kernel_v,
(assignvariableop_102_adam_output3_bias_v.
*assignvariableop_103_adam_output4_kernel_v,
(assignvariableop_104_adam_output4_bias_v
identity_106¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_100¢AssignVariableOp_101¢AssignVariableOp_102¢AssignVariableOp_103¢AssignVariableOp_104¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_74¢AssignVariableOp_75¢AssignVariableOp_76¢AssignVariableOp_77¢AssignVariableOp_78¢AssignVariableOp_79¢AssignVariableOp_8¢AssignVariableOp_80¢AssignVariableOp_81¢AssignVariableOp_82¢AssignVariableOp_83¢AssignVariableOp_84¢AssignVariableOp_85¢AssignVariableOp_86¢AssignVariableOp_87¢AssignVariableOp_88¢AssignVariableOp_89¢AssignVariableOp_9¢AssignVariableOp_90¢AssignVariableOp_91¢AssignVariableOp_92¢AssignVariableOp_93¢AssignVariableOp_94¢AssignVariableOp_95¢AssignVariableOp_96¢AssignVariableOp_97¢AssignVariableOp_98¢AssignVariableOp_997
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:j*
dtype0*£6
value6B6jB4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-3/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-9/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_nameså
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:j*
dtype0*é
valueßBÜjB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesÀ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*¾
_output_shapes«
¨::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*x
dtypesn
l2j		2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity¢
AssignVariableOpAssignVariableOp#assignvariableop_normalization_meanIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1®
AssignVariableOp_1AssignVariableOp)assignvariableop_1_normalization_varianceIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2«
AssignVariableOp_2AssignVariableOp&assignvariableop_2_normalization_countIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¨
AssignVariableOp_3AssignVariableOp#assignvariableop_3_stem_conv_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¥
AssignVariableOp_4AssignVariableOp assignvariableop_4_stem_bn_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¤
AssignVariableOp_5AssignVariableOpassignvariableop_5_stem_bn_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6«
AssignVariableOp_6AssignVariableOp&assignvariableop_6_stem_bn_moving_meanIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¯
AssignVariableOp_7AssignVariableOp*assignvariableop_7_stem_bn_moving_varianceIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8·
AssignVariableOp_8AssignVariableOp2assignvariableop_8_block1a_dwconv_depthwise_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¨
AssignVariableOp_9AssignVariableOp#assignvariableop_9_block1a_bn_gammaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10«
AssignVariableOp_10AssignVariableOp#assignvariableop_10_block1a_bn_betaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11²
AssignVariableOp_11AssignVariableOp*assignvariableop_11_block1a_bn_moving_meanIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12¶
AssignVariableOp_12AssignVariableOp.assignvariableop_12_block1a_bn_moving_varianceIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13´
AssignVariableOp_13AssignVariableOp,assignvariableop_13_block1a_se_reduce_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14²
AssignVariableOp_14AssignVariableOp*assignvariableop_14_block1a_se_reduce_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15´
AssignVariableOp_15AssignVariableOp,assignvariableop_15_block1a_se_expand_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16²
AssignVariableOp_16AssignVariableOp*assignvariableop_16_block1a_se_expand_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17·
AssignVariableOp_17AssignVariableOp/assignvariableop_17_block1a_project_conv_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18´
AssignVariableOp_18AssignVariableOp,assignvariableop_18_block1a_project_bn_gammaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19³
AssignVariableOp_19AssignVariableOp+assignvariableop_19_block1a_project_bn_betaIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20º
AssignVariableOp_20AssignVariableOp2assignvariableop_20_block1a_project_bn_moving_meanIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21¾
AssignVariableOp_21AssignVariableOp6assignvariableop_21_block1a_project_bn_moving_varianceIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22»
AssignVariableOp_22AssignVariableOp3assignvariableop_22_block1b_dwconv_depthwise_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23¬
AssignVariableOp_23AssignVariableOp$assignvariableop_23_block1b_bn_gammaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24«
AssignVariableOp_24AssignVariableOp#assignvariableop_24_block1b_bn_betaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25²
AssignVariableOp_25AssignVariableOp*assignvariableop_25_block1b_bn_moving_meanIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26¶
AssignVariableOp_26AssignVariableOp.assignvariableop_26_block1b_bn_moving_varianceIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27´
AssignVariableOp_27AssignVariableOp,assignvariableop_27_block1b_se_reduce_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28²
AssignVariableOp_28AssignVariableOp*assignvariableop_28_block1b_se_reduce_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29´
AssignVariableOp_29AssignVariableOp,assignvariableop_29_block1b_se_expand_kernelIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30²
AssignVariableOp_30AssignVariableOp*assignvariableop_30_block1b_se_expand_biasIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31·
AssignVariableOp_31AssignVariableOp/assignvariableop_31_block1b_project_conv_kernelIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32´
AssignVariableOp_32AssignVariableOp,assignvariableop_32_block1b_project_bn_gammaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33³
AssignVariableOp_33AssignVariableOp+assignvariableop_33_block1b_project_bn_betaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34º
AssignVariableOp_34AssignVariableOp2assignvariableop_34_block1b_project_bn_moving_meanIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35¾
AssignVariableOp_35AssignVariableOp6assignvariableop_35_block1b_project_bn_moving_varianceIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36©
AssignVariableOp_36AssignVariableOp!assignvariableop_36_conv2d_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37§
AssignVariableOp_37AssignVariableOpassignvariableop_37_conv2d_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38¨
AssignVariableOp_38AssignVariableOp assignvariableop_38_dense_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39¦
AssignVariableOp_39AssignVariableOpassignvariableop_39_dense_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40ª
AssignVariableOp_40AssignVariableOp"assignvariableop_40_dense_1_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41¨
AssignVariableOp_41AssignVariableOp assignvariableop_41_dense_1_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42ª
AssignVariableOp_42AssignVariableOp"assignvariableop_42_dense_2_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43¨
AssignVariableOp_43AssignVariableOp assignvariableop_43_dense_2_biasIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44ª
AssignVariableOp_44AssignVariableOp"assignvariableop_44_dense_3_kernelIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45¨
AssignVariableOp_45AssignVariableOp assignvariableop_45_dense_3_biasIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46ª
AssignVariableOp_46AssignVariableOp"assignvariableop_46_output1_kernelIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47¨
AssignVariableOp_47AssignVariableOp assignvariableop_47_output1_biasIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48ª
AssignVariableOp_48AssignVariableOp"assignvariableop_48_output2_kernelIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49¨
AssignVariableOp_49AssignVariableOp assignvariableop_49_output2_biasIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50ª
AssignVariableOp_50AssignVariableOp"assignvariableop_50_output3_kernelIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51¨
AssignVariableOp_51AssignVariableOp assignvariableop_51_output3_biasIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52ª
AssignVariableOp_52AssignVariableOp"assignvariableop_52_output4_kernelIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53¨
AssignVariableOp_53AssignVariableOp assignvariableop_53_output4_biasIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_54¥
AssignVariableOp_54AssignVariableOpassignvariableop_54_adam_iterIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55§
AssignVariableOp_55AssignVariableOpassignvariableop_55_adam_beta_1Identity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56§
AssignVariableOp_56AssignVariableOpassignvariableop_56_adam_beta_2Identity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57¦
AssignVariableOp_57AssignVariableOpassignvariableop_57_adam_decayIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58®
AssignVariableOp_58AssignVariableOp&assignvariableop_58_adam_learning_rateIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59¡
AssignVariableOp_59AssignVariableOpassignvariableop_59_totalIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60¡
AssignVariableOp_60AssignVariableOpassignvariableop_60_countIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61£
AssignVariableOp_61AssignVariableOpassignvariableop_61_total_1Identity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62£
AssignVariableOp_62AssignVariableOpassignvariableop_62_count_1Identity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63£
AssignVariableOp_63AssignVariableOpassignvariableop_63_total_2Identity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64£
AssignVariableOp_64AssignVariableOpassignvariableop_64_count_2Identity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65£
AssignVariableOp_65AssignVariableOpassignvariableop_65_total_3Identity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66£
AssignVariableOp_66AssignVariableOpassignvariableop_66_count_3Identity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67£
AssignVariableOp_67AssignVariableOpassignvariableop_67_total_4Identity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68£
AssignVariableOp_68AssignVariableOpassignvariableop_68_count_4Identity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69°
AssignVariableOp_69AssignVariableOp(assignvariableop_69_adam_conv2d_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70®
AssignVariableOp_70AssignVariableOp&assignvariableop_70_adam_conv2d_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71¯
AssignVariableOp_71AssignVariableOp'assignvariableop_71_adam_dense_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72­
AssignVariableOp_72AssignVariableOp%assignvariableop_72_adam_dense_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73±
AssignVariableOp_73AssignVariableOp)assignvariableop_73_adam_dense_1_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74¯
AssignVariableOp_74AssignVariableOp'assignvariableop_74_adam_dense_1_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75±
AssignVariableOp_75AssignVariableOp)assignvariableop_75_adam_dense_2_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76¯
AssignVariableOp_76AssignVariableOp'assignvariableop_76_adam_dense_2_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77±
AssignVariableOp_77AssignVariableOp)assignvariableop_77_adam_dense_3_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78¯
AssignVariableOp_78AssignVariableOp'assignvariableop_78_adam_dense_3_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79±
AssignVariableOp_79AssignVariableOp)assignvariableop_79_adam_output1_kernel_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80¯
AssignVariableOp_80AssignVariableOp'assignvariableop_80_adam_output1_bias_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81±
AssignVariableOp_81AssignVariableOp)assignvariableop_81_adam_output2_kernel_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82¯
AssignVariableOp_82AssignVariableOp'assignvariableop_82_adam_output2_bias_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83±
AssignVariableOp_83AssignVariableOp)assignvariableop_83_adam_output3_kernel_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84¯
AssignVariableOp_84AssignVariableOp'assignvariableop_84_adam_output3_bias_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85±
AssignVariableOp_85AssignVariableOp)assignvariableop_85_adam_output4_kernel_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86¯
AssignVariableOp_86AssignVariableOp'assignvariableop_86_adam_output4_bias_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87°
AssignVariableOp_87AssignVariableOp(assignvariableop_87_adam_conv2d_kernel_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88®
AssignVariableOp_88AssignVariableOp&assignvariableop_88_adam_conv2d_bias_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89¯
AssignVariableOp_89AssignVariableOp'assignvariableop_89_adam_dense_kernel_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90­
AssignVariableOp_90AssignVariableOp%assignvariableop_90_adam_dense_bias_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91±
AssignVariableOp_91AssignVariableOp)assignvariableop_91_adam_dense_1_kernel_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92¯
AssignVariableOp_92AssignVariableOp'assignvariableop_92_adam_dense_1_bias_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93±
AssignVariableOp_93AssignVariableOp)assignvariableop_93_adam_dense_2_kernel_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94¯
AssignVariableOp_94AssignVariableOp'assignvariableop_94_adam_dense_2_bias_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95±
AssignVariableOp_95AssignVariableOp)assignvariableop_95_adam_dense_3_kernel_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96¯
AssignVariableOp_96AssignVariableOp'assignvariableop_96_adam_dense_3_bias_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97±
AssignVariableOp_97AssignVariableOp)assignvariableop_97_adam_output1_kernel_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98¯
AssignVariableOp_98AssignVariableOp'assignvariableop_98_adam_output1_bias_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99±
AssignVariableOp_99AssignVariableOp)assignvariableop_99_adam_output2_kernel_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100³
AssignVariableOp_100AssignVariableOp(assignvariableop_100_adam_output2_bias_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_100q
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:2
Identity_101µ
AssignVariableOp_101AssignVariableOp*assignvariableop_101_adam_output3_kernel_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_101q
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:2
Identity_102³
AssignVariableOp_102AssignVariableOp(assignvariableop_102_adam_output3_bias_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_102q
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:2
Identity_103µ
AssignVariableOp_103AssignVariableOp*assignvariableop_103_adam_output4_kernel_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_103q
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:2
Identity_104³
AssignVariableOp_104AssignVariableOp(assignvariableop_104_adam_output4_bias_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1049
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpë
Identity_105Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_105ß
Identity_106IdentityIdentity_105:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
T0*
_output_shapes
: 2
Identity_106"%
identity_106Identity_106:output:0*»
_input_shapes©
¦: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
	
Ü
C__inference_output3_layer_call_and_return_conditional_losses_102284

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó
f
H__inference_block1b_drop_layer_call_and_return_conditional_losses_102104

inputs

identity_1d
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identitys

Identity_1IdentityIdentity:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
ñ
N__inference_block1a_project_bn_layer_call_and_return_conditional_losses_101638

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3ì
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Á¯
*
 __inference__wrapped_model_97997
input_17
3model_normalization_reshape_readvariableop_resource9
5model_normalization_reshape_1_readvariableop_resource2
.model_stem_conv_conv2d_readvariableop_resource)
%model_stem_bn_readvariableop_resource+
'model_stem_bn_readvariableop_1_resource:
6model_stem_bn_fusedbatchnormv3_readvariableop_resource<
8model_stem_bn_fusedbatchnormv3_readvariableop_1_resource:
6model_block1a_dwconv_depthwise_readvariableop_resource,
(model_block1a_bn_readvariableop_resource.
*model_block1a_bn_readvariableop_1_resource=
9model_block1a_bn_fusedbatchnormv3_readvariableop_resource?
;model_block1a_bn_fusedbatchnormv3_readvariableop_1_resource:
6model_block1a_se_reduce_conv2d_readvariableop_resource;
7model_block1a_se_reduce_biasadd_readvariableop_resource:
6model_block1a_se_expand_conv2d_readvariableop_resource;
7model_block1a_se_expand_biasadd_readvariableop_resource=
9model_block1a_project_conv_conv2d_readvariableop_resource4
0model_block1a_project_bn_readvariableop_resource6
2model_block1a_project_bn_readvariableop_1_resourceE
Amodel_block1a_project_bn_fusedbatchnormv3_readvariableop_resourceG
Cmodel_block1a_project_bn_fusedbatchnormv3_readvariableop_1_resource:
6model_block1b_dwconv_depthwise_readvariableop_resource,
(model_block1b_bn_readvariableop_resource.
*model_block1b_bn_readvariableop_1_resource=
9model_block1b_bn_fusedbatchnormv3_readvariableop_resource?
;model_block1b_bn_fusedbatchnormv3_readvariableop_1_resource:
6model_block1b_se_reduce_conv2d_readvariableop_resource;
7model_block1b_se_reduce_biasadd_readvariableop_resource:
6model_block1b_se_expand_conv2d_readvariableop_resource;
7model_block1b_se_expand_biasadd_readvariableop_resource=
9model_block1b_project_conv_conv2d_readvariableop_resource4
0model_block1b_project_bn_readvariableop_resource6
2model_block1b_project_bn_readvariableop_1_resourceE
Amodel_block1b_project_bn_fusedbatchnormv3_readvariableop_resourceG
Cmodel_block1b_project_bn_fusedbatchnormv3_readvariableop_1_resource/
+model_conv2d_conv2d_readvariableop_resource0
,model_conv2d_biasadd_readvariableop_resource0
,model_dense_3_matmul_readvariableop_resource1
-model_dense_3_biasadd_readvariableop_resource0
,model_dense_2_matmul_readvariableop_resource1
-model_dense_2_biasadd_readvariableop_resource0
,model_dense_1_matmul_readvariableop_resource1
-model_dense_1_biasadd_readvariableop_resource.
*model_dense_matmul_readvariableop_resource/
+model_dense_biasadd_readvariableop_resource0
,model_output4_matmul_readvariableop_resource1
-model_output4_biasadd_readvariableop_resource0
,model_output3_matmul_readvariableop_resource1
-model_output3_biasadd_readvariableop_resource0
,model_output2_matmul_readvariableop_resource1
-model_output2_biasadd_readvariableop_resource0
,model_output1_matmul_readvariableop_resource1
-model_output1_biasadd_readvariableop_resource
identity

identity_1

identity_2

identity_3¢0model/block1a_bn/FusedBatchNormV3/ReadVariableOp¢2model/block1a_bn/FusedBatchNormV3/ReadVariableOp_1¢model/block1a_bn/ReadVariableOp¢!model/block1a_bn/ReadVariableOp_1¢-model/block1a_dwconv/depthwise/ReadVariableOp¢8model/block1a_project_bn/FusedBatchNormV3/ReadVariableOp¢:model/block1a_project_bn/FusedBatchNormV3/ReadVariableOp_1¢'model/block1a_project_bn/ReadVariableOp¢)model/block1a_project_bn/ReadVariableOp_1¢0model/block1a_project_conv/Conv2D/ReadVariableOp¢.model/block1a_se_expand/BiasAdd/ReadVariableOp¢-model/block1a_se_expand/Conv2D/ReadVariableOp¢.model/block1a_se_reduce/BiasAdd/ReadVariableOp¢-model/block1a_se_reduce/Conv2D/ReadVariableOp¢0model/block1b_bn/FusedBatchNormV3/ReadVariableOp¢2model/block1b_bn/FusedBatchNormV3/ReadVariableOp_1¢model/block1b_bn/ReadVariableOp¢!model/block1b_bn/ReadVariableOp_1¢-model/block1b_dwconv/depthwise/ReadVariableOp¢8model/block1b_project_bn/FusedBatchNormV3/ReadVariableOp¢:model/block1b_project_bn/FusedBatchNormV3/ReadVariableOp_1¢'model/block1b_project_bn/ReadVariableOp¢)model/block1b_project_bn/ReadVariableOp_1¢0model/block1b_project_conv/Conv2D/ReadVariableOp¢.model/block1b_se_expand/BiasAdd/ReadVariableOp¢-model/block1b_se_expand/Conv2D/ReadVariableOp¢.model/block1b_se_reduce/BiasAdd/ReadVariableOp¢-model/block1b_se_reduce/Conv2D/ReadVariableOp¢#model/conv2d/BiasAdd/ReadVariableOp¢"model/conv2d/Conv2D/ReadVariableOp¢"model/dense/BiasAdd/ReadVariableOp¢!model/dense/MatMul/ReadVariableOp¢$model/dense_1/BiasAdd/ReadVariableOp¢#model/dense_1/MatMul/ReadVariableOp¢$model/dense_2/BiasAdd/ReadVariableOp¢#model/dense_2/MatMul/ReadVariableOp¢$model/dense_3/BiasAdd/ReadVariableOp¢#model/dense_3/MatMul/ReadVariableOp¢*model/normalization/Reshape/ReadVariableOp¢,model/normalization/Reshape_1/ReadVariableOp¢$model/output1/BiasAdd/ReadVariableOp¢#model/output1/MatMul/ReadVariableOp¢$model/output2/BiasAdd/ReadVariableOp¢#model/output2/MatMul/ReadVariableOp¢$model/output3/BiasAdd/ReadVariableOp¢#model/output3/MatMul/ReadVariableOp¢$model/output4/BiasAdd/ReadVariableOp¢#model/output4/MatMul/ReadVariableOp¢-model/stem_bn/FusedBatchNormV3/ReadVariableOp¢/model/stem_bn/FusedBatchNormV3/ReadVariableOp_1¢model/stem_bn/ReadVariableOp¢model/stem_bn/ReadVariableOp_1¢%model/stem_conv/Conv2D/ReadVariableOpu
model/rescaling/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;2
model/rescaling/Cast/xy
model/rescaling/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model/rescaling/Cast_1/x
model/rescaling/mulMulinput_1model/rescaling/Cast/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬2
model/rescaling/mul«
model/rescaling/addAddV2model/rescaling/mul:z:0!model/rescaling/Cast_1/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬2
model/rescaling/addÈ
*model/normalization/Reshape/ReadVariableOpReadVariableOp3model_normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02,
*model/normalization/Reshape/ReadVariableOp
!model/normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2#
!model/normalization/Reshape/shapeÖ
model/normalization/ReshapeReshape2model/normalization/Reshape/ReadVariableOp:value:0*model/normalization/Reshape/shape:output:0*
T0*&
_output_shapes
:2
model/normalization/ReshapeÎ
,model/normalization/Reshape_1/ReadVariableOpReadVariableOp5model_normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,model/normalization/Reshape_1/ReadVariableOp£
#model/normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2%
#model/normalization/Reshape_1/shapeÞ
model/normalization/Reshape_1Reshape4model/normalization/Reshape_1/ReadVariableOp:value:0,model/normalization/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
model/normalization/Reshape_1´
model/normalization/subSubmodel/rescaling/add:z:0$model/normalization/Reshape:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬2
model/normalization/sub
model/normalization/SqrtSqrt&model/normalization/Reshape_1:output:0*
T0*&
_output_shapes
:2
model/normalization/Sqrt
model/normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö32
model/normalization/Maximum/y¼
model/normalization/MaximumMaximummodel/normalization/Sqrt:y:0&model/normalization/Maximum/y:output:0*
T0*&
_output_shapes
:2
model/normalization/Maximum¿
model/normalization/truedivRealDivmodel/normalization/sub:z:0model/normalization/Maximum:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬2
model/normalization/truedivµ
 model/stem_conv_pad/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                               2"
 model/stem_conv_pad/Pad/paddingsÁ
model/stem_conv_pad/PadPadmodel/normalization/truediv:z:0)model/stem_conv_pad/Pad/paddings:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­­2
model/stem_conv_pad/PadÅ
%model/stem_conv/Conv2D/ReadVariableOpReadVariableOp.model_stem_conv_conv2d_readvariableop_resource*&
_output_shapes
:(*
dtype02'
%model/stem_conv/Conv2D/ReadVariableOpð
model/stem_conv/Conv2DConv2D model/stem_conv_pad/Pad:output:0-model/stem_conv/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
paddingVALID*
strides
2
model/stem_conv/Conv2D
model/stem_bn/ReadVariableOpReadVariableOp%model_stem_bn_readvariableop_resource*
_output_shapes
:(*
dtype02
model/stem_bn/ReadVariableOp¤
model/stem_bn/ReadVariableOp_1ReadVariableOp'model_stem_bn_readvariableop_1_resource*
_output_shapes
:(*
dtype02 
model/stem_bn/ReadVariableOp_1Ñ
-model/stem_bn/FusedBatchNormV3/ReadVariableOpReadVariableOp6model_stem_bn_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:(*
dtype02/
-model/stem_bn/FusedBatchNormV3/ReadVariableOp×
/model/stem_bn/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp8model_stem_bn_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:(*
dtype021
/model/stem_bn/FusedBatchNormV3/ReadVariableOp_1¹
model/stem_bn/FusedBatchNormV3FusedBatchNormV3model/stem_conv/Conv2D:output:0$model/stem_bn/ReadVariableOp:value:0&model/stem_bn/ReadVariableOp_1:value:05model/stem_bn/FusedBatchNormV3/ReadVariableOp:value:07model/stem_bn/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ(:(:(:(:(:*
epsilon%o:*
is_training( 2 
model/stem_bn/FusedBatchNormV3©
model/stem_activation/SigmoidSigmoid"model/stem_bn/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
model/stem_activation/SigmoidÀ
model/stem_activation/mulMul"model/stem_bn/FusedBatchNormV3:y:0!model/stem_activation/Sigmoid:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
model/stem_activation/mul§
model/stem_activation/IdentityIdentitymodel/stem_activation/mul:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2 
model/stem_activation/Identity
model/stem_activation/IdentityN	IdentityNmodel/stem_activation/mul:z:0"model/stem_bn/FusedBatchNormV3:y:0*
T
2*+
_gradient_op_typeCustomGradient-97775*N
_output_shapes<
::ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(2!
model/stem_activation/IdentityNÝ
-model/block1a_dwconv/depthwise/ReadVariableOpReadVariableOp6model_block1a_dwconv_depthwise_readvariableop_resource*&
_output_shapes
:(*
dtype02/
-model/block1a_dwconv/depthwise/ReadVariableOp¥
$model/block1a_dwconv/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      (      2&
$model/block1a_dwconv/depthwise/Shape­
,model/block1a_dwconv/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2.
,model/block1a_dwconv/depthwise/dilation_rate
model/block1a_dwconv/depthwiseDepthwiseConv2dNative(model/stem_activation/IdentityN:output:05model/block1a_dwconv/depthwise/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
paddingSAME*
strides
2 
model/block1a_dwconv/depthwise§
model/block1a_bn/ReadVariableOpReadVariableOp(model_block1a_bn_readvariableop_resource*
_output_shapes
:(*
dtype02!
model/block1a_bn/ReadVariableOp­
!model/block1a_bn/ReadVariableOp_1ReadVariableOp*model_block1a_bn_readvariableop_1_resource*
_output_shapes
:(*
dtype02#
!model/block1a_bn/ReadVariableOp_1Ú
0model/block1a_bn/FusedBatchNormV3/ReadVariableOpReadVariableOp9model_block1a_bn_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:(*
dtype022
0model/block1a_bn/FusedBatchNormV3/ReadVariableOpà
2model/block1a_bn/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;model_block1a_bn_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:(*
dtype024
2model/block1a_bn/FusedBatchNormV3/ReadVariableOp_1Ó
!model/block1a_bn/FusedBatchNormV3FusedBatchNormV3'model/block1a_dwconv/depthwise:output:0'model/block1a_bn/ReadVariableOp:value:0)model/block1a_bn/ReadVariableOp_1:value:08model/block1a_bn/FusedBatchNormV3/ReadVariableOp:value:0:model/block1a_bn/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ(:(:(:(:(:*
epsilon%o:*
is_training( 2#
!model/block1a_bn/FusedBatchNormV3²
 model/block1a_activation/SigmoidSigmoid%model/block1a_bn/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2"
 model/block1a_activation/SigmoidÌ
model/block1a_activation/mulMul%model/block1a_bn/FusedBatchNormV3:y:0$model/block1a_activation/Sigmoid:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
model/block1a_activation/mul°
!model/block1a_activation/IdentityIdentity model/block1a_activation/mul:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2#
!model/block1a_activation/Identity¨
"model/block1a_activation/IdentityN	IdentityN model/block1a_activation/mul:z:0%model/block1a_bn/FusedBatchNormV3:y:0*
T
2*+
_gradient_op_typeCustomGradient-97800*N
_output_shapes<
::ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(2$
"model/block1a_activation/IdentityN³
/model/block1a_se_squeeze/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      21
/model/block1a_se_squeeze/Mean/reduction_indicesß
model/block1a_se_squeeze/MeanMean+model/block1a_activation/IdentityN:output:08model/block1a_se_squeeze/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
model/block1a_se_squeeze/Mean
model/block1a_se_reshape/ShapeShape&model/block1a_se_squeeze/Mean:output:0*
T0*
_output_shapes
:2 
model/block1a_se_reshape/Shape¦
,model/block1a_se_reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,model/block1a_se_reshape/strided_slice/stackª
.model/block1a_se_reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.model/block1a_se_reshape/strided_slice/stack_1ª
.model/block1a_se_reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.model/block1a_se_reshape/strided_slice/stack_2ø
&model/block1a_se_reshape/strided_sliceStridedSlice'model/block1a_se_reshape/Shape:output:05model/block1a_se_reshape/strided_slice/stack:output:07model/block1a_se_reshape/strided_slice/stack_1:output:07model/block1a_se_reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&model/block1a_se_reshape/strided_slice
(model/block1a_se_reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(model/block1a_se_reshape/Reshape/shape/1
(model/block1a_se_reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(model/block1a_se_reshape/Reshape/shape/2
(model/block1a_se_reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :(2*
(model/block1a_se_reshape/Reshape/shape/3Ð
&model/block1a_se_reshape/Reshape/shapePack/model/block1a_se_reshape/strided_slice:output:01model/block1a_se_reshape/Reshape/shape/1:output:01model/block1a_se_reshape/Reshape/shape/2:output:01model/block1a_se_reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2(
&model/block1a_se_reshape/Reshape/shapeâ
 model/block1a_se_reshape/ReshapeReshape&model/block1a_se_squeeze/Mean:output:0/model/block1a_se_reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2"
 model/block1a_se_reshape/ReshapeÝ
-model/block1a_se_reduce/Conv2D/ReadVariableOpReadVariableOp6model_block1a_se_reduce_conv2d_readvariableop_resource*&
_output_shapes
:(
*
dtype02/
-model/block1a_se_reduce/Conv2D/ReadVariableOp
model/block1a_se_reduce/Conv2DConv2D)model/block1a_se_reshape/Reshape:output:05model/block1a_se_reduce/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
paddingSAME*
strides
2 
model/block1a_se_reduce/Conv2DÔ
.model/block1a_se_reduce/BiasAdd/ReadVariableOpReadVariableOp7model_block1a_se_reduce_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype020
.model/block1a_se_reduce/BiasAdd/ReadVariableOpè
model/block1a_se_reduce/BiasAddBiasAdd'model/block1a_se_reduce/Conv2D:output:06model/block1a_se_reduce/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2!
model/block1a_se_reduce/BiasAdd±
model/block1a_se_reduce/SigmoidSigmoid(model/block1a_se_reduce/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2!
model/block1a_se_reduce/SigmoidÊ
model/block1a_se_reduce/mulMul(model/block1a_se_reduce/BiasAdd:output:0#model/block1a_se_reduce/Sigmoid:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
model/block1a_se_reduce/mul«
 model/block1a_se_reduce/IdentityIdentitymodel/block1a_se_reduce/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2"
 model/block1a_se_reduce/Identity¤
!model/block1a_se_reduce/IdentityN	IdentityNmodel/block1a_se_reduce/mul:z:0(model/block1a_se_reduce/BiasAdd:output:0*
T
2*+
_gradient_op_typeCustomGradient-97824*J
_output_shapes8
6:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
2#
!model/block1a_se_reduce/IdentityNÝ
-model/block1a_se_expand/Conv2D/ReadVariableOpReadVariableOp6model_block1a_se_expand_conv2d_readvariableop_resource*&
_output_shapes
:
(*
dtype02/
-model/block1a_se_expand/Conv2D/ReadVariableOp
model/block1a_se_expand/Conv2DConv2D*model/block1a_se_reduce/IdentityN:output:05model/block1a_se_expand/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
paddingSAME*
strides
2 
model/block1a_se_expand/Conv2DÔ
.model/block1a_se_expand/BiasAdd/ReadVariableOpReadVariableOp7model_block1a_se_expand_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype020
.model/block1a_se_expand/BiasAdd/ReadVariableOpè
model/block1a_se_expand/BiasAddBiasAdd'model/block1a_se_expand/Conv2D:output:06model/block1a_se_expand/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2!
model/block1a_se_expand/BiasAdd±
model/block1a_se_expand/SigmoidSigmoid(model/block1a_se_expand/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2!
model/block1a_se_expand/SigmoidÏ
model/block1a_se_excite/mulMul+model/block1a_activation/IdentityN:output:0#model/block1a_se_expand/Sigmoid:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
model/block1a_se_excite/mulæ
0model/block1a_project_conv/Conv2D/ReadVariableOpReadVariableOp9model_block1a_project_conv_conv2d_readvariableop_resource*&
_output_shapes
:(*
dtype022
0model/block1a_project_conv/Conv2D/ReadVariableOp
!model/block1a_project_conv/Conv2DConv2Dmodel/block1a_se_excite/mul:z:08model/block1a_project_conv/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2#
!model/block1a_project_conv/Conv2D¿
'model/block1a_project_bn/ReadVariableOpReadVariableOp0model_block1a_project_bn_readvariableop_resource*
_output_shapes
:*
dtype02)
'model/block1a_project_bn/ReadVariableOpÅ
)model/block1a_project_bn/ReadVariableOp_1ReadVariableOp2model_block1a_project_bn_readvariableop_1_resource*
_output_shapes
:*
dtype02+
)model/block1a_project_bn/ReadVariableOp_1ò
8model/block1a_project_bn/FusedBatchNormV3/ReadVariableOpReadVariableOpAmodel_block1a_project_bn_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02:
8model/block1a_project_bn/FusedBatchNormV3/ReadVariableOpø
:model/block1a_project_bn/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpCmodel_block1a_project_bn_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02<
:model/block1a_project_bn/FusedBatchNormV3/ReadVariableOp_1
)model/block1a_project_bn/FusedBatchNormV3FusedBatchNormV3*model/block1a_project_conv/Conv2D:output:0/model/block1a_project_bn/ReadVariableOp:value:01model/block1a_project_bn/ReadVariableOp_1:value:0@model/block1a_project_bn/FusedBatchNormV3/ReadVariableOp:value:0Bmodel/block1a_project_bn/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2+
)model/block1a_project_bn/FusedBatchNormV3Ý
-model/block1b_dwconv/depthwise/ReadVariableOpReadVariableOp6model_block1b_dwconv_depthwise_readvariableop_resource*&
_output_shapes
:*
dtype02/
-model/block1b_dwconv/depthwise/ReadVariableOp¥
$model/block1b_dwconv/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2&
$model/block1b_dwconv/depthwise/Shape­
,model/block1b_dwconv/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2.
,model/block1b_dwconv/depthwise/dilation_rate£
model/block1b_dwconv/depthwiseDepthwiseConv2dNative-model/block1a_project_bn/FusedBatchNormV3:y:05model/block1b_dwconv/depthwise/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2 
model/block1b_dwconv/depthwise§
model/block1b_bn/ReadVariableOpReadVariableOp(model_block1b_bn_readvariableop_resource*
_output_shapes
:*
dtype02!
model/block1b_bn/ReadVariableOp­
!model/block1b_bn/ReadVariableOp_1ReadVariableOp*model_block1b_bn_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!model/block1b_bn/ReadVariableOp_1Ú
0model/block1b_bn/FusedBatchNormV3/ReadVariableOpReadVariableOp9model_block1b_bn_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype022
0model/block1b_bn/FusedBatchNormV3/ReadVariableOpà
2model/block1b_bn/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;model_block1b_bn_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype024
2model/block1b_bn/FusedBatchNormV3/ReadVariableOp_1Ó
!model/block1b_bn/FusedBatchNormV3FusedBatchNormV3'model/block1b_dwconv/depthwise:output:0'model/block1b_bn/ReadVariableOp:value:0)model/block1b_bn/ReadVariableOp_1:value:08model/block1b_bn/FusedBatchNormV3/ReadVariableOp:value:0:model/block1b_bn/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2#
!model/block1b_bn/FusedBatchNormV3²
 model/block1b_activation/SigmoidSigmoid%model/block1b_bn/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 model/block1b_activation/SigmoidÌ
model/block1b_activation/mulMul%model/block1b_bn/FusedBatchNormV3:y:0$model/block1b_activation/Sigmoid:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/block1b_activation/mul°
!model/block1b_activation/IdentityIdentity model/block1b_activation/mul:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!model/block1b_activation/Identity¨
"model/block1b_activation/IdentityN	IdentityN model/block1b_activation/mul:z:0%model/block1b_bn/FusedBatchNormV3:y:0*
T
2*+
_gradient_op_typeCustomGradient-97874*N
_output_shapes<
::ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ2$
"model/block1b_activation/IdentityN³
/model/block1b_se_squeeze/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      21
/model/block1b_se_squeeze/Mean/reduction_indicesß
model/block1b_se_squeeze/MeanMean+model/block1b_activation/IdentityN:output:08model/block1b_se_squeeze/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/block1b_se_squeeze/Mean
model/block1b_se_reshape/ShapeShape&model/block1b_se_squeeze/Mean:output:0*
T0*
_output_shapes
:2 
model/block1b_se_reshape/Shape¦
,model/block1b_se_reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,model/block1b_se_reshape/strided_slice/stackª
.model/block1b_se_reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.model/block1b_se_reshape/strided_slice/stack_1ª
.model/block1b_se_reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.model/block1b_se_reshape/strided_slice/stack_2ø
&model/block1b_se_reshape/strided_sliceStridedSlice'model/block1b_se_reshape/Shape:output:05model/block1b_se_reshape/strided_slice/stack:output:07model/block1b_se_reshape/strided_slice/stack_1:output:07model/block1b_se_reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&model/block1b_se_reshape/strided_slice
(model/block1b_se_reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(model/block1b_se_reshape/Reshape/shape/1
(model/block1b_se_reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(model/block1b_se_reshape/Reshape/shape/2
(model/block1b_se_reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2*
(model/block1b_se_reshape/Reshape/shape/3Ð
&model/block1b_se_reshape/Reshape/shapePack/model/block1b_se_reshape/strided_slice:output:01model/block1b_se_reshape/Reshape/shape/1:output:01model/block1b_se_reshape/Reshape/shape/2:output:01model/block1b_se_reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2(
&model/block1b_se_reshape/Reshape/shapeâ
 model/block1b_se_reshape/ReshapeReshape&model/block1b_se_squeeze/Mean:output:0/model/block1b_se_reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 model/block1b_se_reshape/ReshapeÝ
-model/block1b_se_reduce/Conv2D/ReadVariableOpReadVariableOp6model_block1b_se_reduce_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02/
-model/block1b_se_reduce/Conv2D/ReadVariableOp
model/block1b_se_reduce/Conv2DConv2D)model/block1b_se_reshape/Reshape:output:05model/block1b_se_reduce/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2 
model/block1b_se_reduce/Conv2DÔ
.model/block1b_se_reduce/BiasAdd/ReadVariableOpReadVariableOp7model_block1b_se_reduce_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.model/block1b_se_reduce/BiasAdd/ReadVariableOpè
model/block1b_se_reduce/BiasAddBiasAdd'model/block1b_se_reduce/Conv2D:output:06model/block1b_se_reduce/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
model/block1b_se_reduce/BiasAdd±
model/block1b_se_reduce/SigmoidSigmoid(model/block1b_se_reduce/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
model/block1b_se_reduce/SigmoidÊ
model/block1b_se_reduce/mulMul(model/block1b_se_reduce/BiasAdd:output:0#model/block1b_se_reduce/Sigmoid:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/block1b_se_reduce/mul«
 model/block1b_se_reduce/IdentityIdentitymodel/block1b_se_reduce/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 model/block1b_se_reduce/Identity¤
!model/block1b_se_reduce/IdentityN	IdentityNmodel/block1b_se_reduce/mul:z:0(model/block1b_se_reduce/BiasAdd:output:0*
T
2*+
_gradient_op_typeCustomGradient-97898*J
_output_shapes8
6:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ2#
!model/block1b_se_reduce/IdentityNÝ
-model/block1b_se_expand/Conv2D/ReadVariableOpReadVariableOp6model_block1b_se_expand_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02/
-model/block1b_se_expand/Conv2D/ReadVariableOp
model/block1b_se_expand/Conv2DConv2D*model/block1b_se_reduce/IdentityN:output:05model/block1b_se_expand/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2 
model/block1b_se_expand/Conv2DÔ
.model/block1b_se_expand/BiasAdd/ReadVariableOpReadVariableOp7model_block1b_se_expand_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.model/block1b_se_expand/BiasAdd/ReadVariableOpè
model/block1b_se_expand/BiasAddBiasAdd'model/block1b_se_expand/Conv2D:output:06model/block1b_se_expand/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
model/block1b_se_expand/BiasAdd±
model/block1b_se_expand/SigmoidSigmoid(model/block1b_se_expand/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
model/block1b_se_expand/SigmoidÏ
model/block1b_se_excite/mulMul+model/block1b_activation/IdentityN:output:0#model/block1b_se_expand/Sigmoid:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/block1b_se_excite/mulæ
0model/block1b_project_conv/Conv2D/ReadVariableOpReadVariableOp9model_block1b_project_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype022
0model/block1b_project_conv/Conv2D/ReadVariableOp
!model/block1b_project_conv/Conv2DConv2Dmodel/block1b_se_excite/mul:z:08model/block1b_project_conv/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2#
!model/block1b_project_conv/Conv2D¿
'model/block1b_project_bn/ReadVariableOpReadVariableOp0model_block1b_project_bn_readvariableop_resource*
_output_shapes
:*
dtype02)
'model/block1b_project_bn/ReadVariableOpÅ
)model/block1b_project_bn/ReadVariableOp_1ReadVariableOp2model_block1b_project_bn_readvariableop_1_resource*
_output_shapes
:*
dtype02+
)model/block1b_project_bn/ReadVariableOp_1ò
8model/block1b_project_bn/FusedBatchNormV3/ReadVariableOpReadVariableOpAmodel_block1b_project_bn_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02:
8model/block1b_project_bn/FusedBatchNormV3/ReadVariableOpø
:model/block1b_project_bn/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpCmodel_block1b_project_bn_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02<
:model/block1b_project_bn/FusedBatchNormV3/ReadVariableOp_1
)model/block1b_project_bn/FusedBatchNormV3FusedBatchNormV3*model/block1b_project_conv/Conv2D:output:0/model/block1b_project_bn/ReadVariableOp:value:01model/block1b_project_bn/ReadVariableOp_1:value:0@model/block1b_project_bn/FusedBatchNormV3/ReadVariableOp:value:0Bmodel/block1b_project_bn/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2+
)model/block1b_project_bn/FusedBatchNormV3±
model/block1b_drop/IdentityIdentity-model/block1b_project_bn/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/block1b_drop/IdentityÈ
model/block1b_add/addAddV2$model/block1b_drop/Identity:output:0-model/block1a_project_bn/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/block1b_add/addÞ
model/average_pooling2d/AvgPoolAvgPoolmodel/block1b_add/add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2!
model/average_pooling2d/AvgPool¼
"model/conv2d/Conv2D/ReadVariableOpReadVariableOp+model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02$
"model/conv2d/Conv2D/ReadVariableOpí
model/conv2d/Conv2DConv2D(model/average_pooling2d/AvgPool:output:0*model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
model/conv2d/Conv2D³
#model/conv2d/BiasAdd/ReadVariableOpReadVariableOp,model_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#model/conv2d/BiasAdd/ReadVariableOp¼
model/conv2d/BiasAddBiasAddmodel/conv2d/Conv2D:output:0+model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/conv2d/BiasAddÑ
model/max_pooling2d/MaxPoolMaxPoolmodel/conv2d/BiasAdd:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2
model/max_pooling2d/MaxPool{
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
  2
model/flatten/Const°
model/flatten/ReshapeReshape$model/max_pooling2d/MaxPool:output:0model/flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/flatten/Reshape¸
#model/dense_3/MatMul/ReadVariableOpReadVariableOp,model_dense_3_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02%
#model/dense_3/MatMul/ReadVariableOpµ
model/dense_3/MatMulMatMulmodel/flatten/Reshape:output:0+model/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/dense_3/MatMul¶
$model/dense_3/BiasAdd/ReadVariableOpReadVariableOp-model_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/dense_3/BiasAdd/ReadVariableOp¹
model/dense_3/BiasAddBiasAddmodel/dense_3/MatMul:product:0,model/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/dense_3/BiasAdd
model/dense_3/ReluRelumodel/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/dense_3/Relu¸
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02%
#model/dense_2/MatMul/ReadVariableOpµ
model/dense_2/MatMulMatMulmodel/flatten/Reshape:output:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/dense_2/MatMul¶
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/dense_2/BiasAdd/ReadVariableOp¹
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/dense_2/BiasAdd
model/dense_2/ReluRelumodel/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/dense_2/Relu¸
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02%
#model/dense_1/MatMul/ReadVariableOpµ
model/dense_1/MatMulMatMulmodel/flatten/Reshape:output:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/dense_1/MatMul¶
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/dense_1/BiasAdd/ReadVariableOp¹
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/dense_1/BiasAdd
model/dense_1/ReluRelumodel/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/dense_1/Relu²
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02#
!model/dense/MatMul/ReadVariableOp¯
model/dense/MatMulMatMulmodel/flatten/Reshape:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/dense/MatMul°
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"model/dense/BiasAdd/ReadVariableOp±
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/dense/BiasAdd|
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/dense/Relu·
#model/output4/MatMul/ReadVariableOpReadVariableOp,model_output4_matmul_readvariableop_resource*
_output_shapes

:*
dtype02%
#model/output4/MatMul/ReadVariableOp·
model/output4/MatMulMatMul model/dense_3/Relu:activations:0+model/output4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/output4/MatMul¶
$model/output4/BiasAdd/ReadVariableOpReadVariableOp-model_output4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/output4/BiasAdd/ReadVariableOp¹
model/output4/BiasAddBiasAddmodel/output4/MatMul:product:0,model/output4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/output4/BiasAdd·
#model/output3/MatMul/ReadVariableOpReadVariableOp,model_output3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02%
#model/output3/MatMul/ReadVariableOp·
model/output3/MatMulMatMul model/dense_2/Relu:activations:0+model/output3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/output3/MatMul¶
$model/output3/BiasAdd/ReadVariableOpReadVariableOp-model_output3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/output3/BiasAdd/ReadVariableOp¹
model/output3/BiasAddBiasAddmodel/output3/MatMul:product:0,model/output3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/output3/BiasAdd·
#model/output2/MatMul/ReadVariableOpReadVariableOp,model_output2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02%
#model/output2/MatMul/ReadVariableOp·
model/output2/MatMulMatMul model/dense_1/Relu:activations:0+model/output2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/output2/MatMul¶
$model/output2/BiasAdd/ReadVariableOpReadVariableOp-model_output2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/output2/BiasAdd/ReadVariableOp¹
model/output2/BiasAddBiasAddmodel/output2/MatMul:product:0,model/output2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/output2/BiasAdd·
#model/output1/MatMul/ReadVariableOpReadVariableOp,model_output1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02%
#model/output1/MatMul/ReadVariableOpµ
model/output1/MatMulMatMulmodel/dense/Relu:activations:0+model/output1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/output1/MatMul¶
$model/output1/BiasAdd/ReadVariableOpReadVariableOp-model_output1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/output1/BiasAdd/ReadVariableOp¹
model/output1/BiasAddBiasAddmodel/output1/MatMul:product:0,model/output1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/output1/BiasAdd
IdentityIdentitymodel/output1/BiasAdd:output:01^model/block1a_bn/FusedBatchNormV3/ReadVariableOp3^model/block1a_bn/FusedBatchNormV3/ReadVariableOp_1 ^model/block1a_bn/ReadVariableOp"^model/block1a_bn/ReadVariableOp_1.^model/block1a_dwconv/depthwise/ReadVariableOp9^model/block1a_project_bn/FusedBatchNormV3/ReadVariableOp;^model/block1a_project_bn/FusedBatchNormV3/ReadVariableOp_1(^model/block1a_project_bn/ReadVariableOp*^model/block1a_project_bn/ReadVariableOp_11^model/block1a_project_conv/Conv2D/ReadVariableOp/^model/block1a_se_expand/BiasAdd/ReadVariableOp.^model/block1a_se_expand/Conv2D/ReadVariableOp/^model/block1a_se_reduce/BiasAdd/ReadVariableOp.^model/block1a_se_reduce/Conv2D/ReadVariableOp1^model/block1b_bn/FusedBatchNormV3/ReadVariableOp3^model/block1b_bn/FusedBatchNormV3/ReadVariableOp_1 ^model/block1b_bn/ReadVariableOp"^model/block1b_bn/ReadVariableOp_1.^model/block1b_dwconv/depthwise/ReadVariableOp9^model/block1b_project_bn/FusedBatchNormV3/ReadVariableOp;^model/block1b_project_bn/FusedBatchNormV3/ReadVariableOp_1(^model/block1b_project_bn/ReadVariableOp*^model/block1b_project_bn/ReadVariableOp_11^model/block1b_project_conv/Conv2D/ReadVariableOp/^model/block1b_se_expand/BiasAdd/ReadVariableOp.^model/block1b_se_expand/Conv2D/ReadVariableOp/^model/block1b_se_reduce/BiasAdd/ReadVariableOp.^model/block1b_se_reduce/Conv2D/ReadVariableOp$^model/conv2d/BiasAdd/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp%^model/dense_3/BiasAdd/ReadVariableOp$^model/dense_3/MatMul/ReadVariableOp+^model/normalization/Reshape/ReadVariableOp-^model/normalization/Reshape_1/ReadVariableOp%^model/output1/BiasAdd/ReadVariableOp$^model/output1/MatMul/ReadVariableOp%^model/output2/BiasAdd/ReadVariableOp$^model/output2/MatMul/ReadVariableOp%^model/output3/BiasAdd/ReadVariableOp$^model/output3/MatMul/ReadVariableOp%^model/output4/BiasAdd/ReadVariableOp$^model/output4/MatMul/ReadVariableOp.^model/stem_bn/FusedBatchNormV3/ReadVariableOp0^model/stem_bn/FusedBatchNormV3/ReadVariableOp_1^model/stem_bn/ReadVariableOp^model/stem_bn/ReadVariableOp_1&^model/stem_conv/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identitymodel/output2/BiasAdd:output:01^model/block1a_bn/FusedBatchNormV3/ReadVariableOp3^model/block1a_bn/FusedBatchNormV3/ReadVariableOp_1 ^model/block1a_bn/ReadVariableOp"^model/block1a_bn/ReadVariableOp_1.^model/block1a_dwconv/depthwise/ReadVariableOp9^model/block1a_project_bn/FusedBatchNormV3/ReadVariableOp;^model/block1a_project_bn/FusedBatchNormV3/ReadVariableOp_1(^model/block1a_project_bn/ReadVariableOp*^model/block1a_project_bn/ReadVariableOp_11^model/block1a_project_conv/Conv2D/ReadVariableOp/^model/block1a_se_expand/BiasAdd/ReadVariableOp.^model/block1a_se_expand/Conv2D/ReadVariableOp/^model/block1a_se_reduce/BiasAdd/ReadVariableOp.^model/block1a_se_reduce/Conv2D/ReadVariableOp1^model/block1b_bn/FusedBatchNormV3/ReadVariableOp3^model/block1b_bn/FusedBatchNormV3/ReadVariableOp_1 ^model/block1b_bn/ReadVariableOp"^model/block1b_bn/ReadVariableOp_1.^model/block1b_dwconv/depthwise/ReadVariableOp9^model/block1b_project_bn/FusedBatchNormV3/ReadVariableOp;^model/block1b_project_bn/FusedBatchNormV3/ReadVariableOp_1(^model/block1b_project_bn/ReadVariableOp*^model/block1b_project_bn/ReadVariableOp_11^model/block1b_project_conv/Conv2D/ReadVariableOp/^model/block1b_se_expand/BiasAdd/ReadVariableOp.^model/block1b_se_expand/Conv2D/ReadVariableOp/^model/block1b_se_reduce/BiasAdd/ReadVariableOp.^model/block1b_se_reduce/Conv2D/ReadVariableOp$^model/conv2d/BiasAdd/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp%^model/dense_3/BiasAdd/ReadVariableOp$^model/dense_3/MatMul/ReadVariableOp+^model/normalization/Reshape/ReadVariableOp-^model/normalization/Reshape_1/ReadVariableOp%^model/output1/BiasAdd/ReadVariableOp$^model/output1/MatMul/ReadVariableOp%^model/output2/BiasAdd/ReadVariableOp$^model/output2/MatMul/ReadVariableOp%^model/output3/BiasAdd/ReadVariableOp$^model/output3/MatMul/ReadVariableOp%^model/output4/BiasAdd/ReadVariableOp$^model/output4/MatMul/ReadVariableOp.^model/stem_bn/FusedBatchNormV3/ReadVariableOp0^model/stem_bn/FusedBatchNormV3/ReadVariableOp_1^model/stem_bn/ReadVariableOp^model/stem_bn/ReadVariableOp_1&^model/stem_conv/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identitymodel/output3/BiasAdd:output:01^model/block1a_bn/FusedBatchNormV3/ReadVariableOp3^model/block1a_bn/FusedBatchNormV3/ReadVariableOp_1 ^model/block1a_bn/ReadVariableOp"^model/block1a_bn/ReadVariableOp_1.^model/block1a_dwconv/depthwise/ReadVariableOp9^model/block1a_project_bn/FusedBatchNormV3/ReadVariableOp;^model/block1a_project_bn/FusedBatchNormV3/ReadVariableOp_1(^model/block1a_project_bn/ReadVariableOp*^model/block1a_project_bn/ReadVariableOp_11^model/block1a_project_conv/Conv2D/ReadVariableOp/^model/block1a_se_expand/BiasAdd/ReadVariableOp.^model/block1a_se_expand/Conv2D/ReadVariableOp/^model/block1a_se_reduce/BiasAdd/ReadVariableOp.^model/block1a_se_reduce/Conv2D/ReadVariableOp1^model/block1b_bn/FusedBatchNormV3/ReadVariableOp3^model/block1b_bn/FusedBatchNormV3/ReadVariableOp_1 ^model/block1b_bn/ReadVariableOp"^model/block1b_bn/ReadVariableOp_1.^model/block1b_dwconv/depthwise/ReadVariableOp9^model/block1b_project_bn/FusedBatchNormV3/ReadVariableOp;^model/block1b_project_bn/FusedBatchNormV3/ReadVariableOp_1(^model/block1b_project_bn/ReadVariableOp*^model/block1b_project_bn/ReadVariableOp_11^model/block1b_project_conv/Conv2D/ReadVariableOp/^model/block1b_se_expand/BiasAdd/ReadVariableOp.^model/block1b_se_expand/Conv2D/ReadVariableOp/^model/block1b_se_reduce/BiasAdd/ReadVariableOp.^model/block1b_se_reduce/Conv2D/ReadVariableOp$^model/conv2d/BiasAdd/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp%^model/dense_3/BiasAdd/ReadVariableOp$^model/dense_3/MatMul/ReadVariableOp+^model/normalization/Reshape/ReadVariableOp-^model/normalization/Reshape_1/ReadVariableOp%^model/output1/BiasAdd/ReadVariableOp$^model/output1/MatMul/ReadVariableOp%^model/output2/BiasAdd/ReadVariableOp$^model/output2/MatMul/ReadVariableOp%^model/output3/BiasAdd/ReadVariableOp$^model/output3/MatMul/ReadVariableOp%^model/output4/BiasAdd/ReadVariableOp$^model/output4/MatMul/ReadVariableOp.^model/stem_bn/FusedBatchNormV3/ReadVariableOp0^model/stem_bn/FusedBatchNormV3/ReadVariableOp_1^model/stem_bn/ReadVariableOp^model/stem_bn/ReadVariableOp_1&^model/stem_conv/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2

Identity_3Identitymodel/output4/BiasAdd:output:01^model/block1a_bn/FusedBatchNormV3/ReadVariableOp3^model/block1a_bn/FusedBatchNormV3/ReadVariableOp_1 ^model/block1a_bn/ReadVariableOp"^model/block1a_bn/ReadVariableOp_1.^model/block1a_dwconv/depthwise/ReadVariableOp9^model/block1a_project_bn/FusedBatchNormV3/ReadVariableOp;^model/block1a_project_bn/FusedBatchNormV3/ReadVariableOp_1(^model/block1a_project_bn/ReadVariableOp*^model/block1a_project_bn/ReadVariableOp_11^model/block1a_project_conv/Conv2D/ReadVariableOp/^model/block1a_se_expand/BiasAdd/ReadVariableOp.^model/block1a_se_expand/Conv2D/ReadVariableOp/^model/block1a_se_reduce/BiasAdd/ReadVariableOp.^model/block1a_se_reduce/Conv2D/ReadVariableOp1^model/block1b_bn/FusedBatchNormV3/ReadVariableOp3^model/block1b_bn/FusedBatchNormV3/ReadVariableOp_1 ^model/block1b_bn/ReadVariableOp"^model/block1b_bn/ReadVariableOp_1.^model/block1b_dwconv/depthwise/ReadVariableOp9^model/block1b_project_bn/FusedBatchNormV3/ReadVariableOp;^model/block1b_project_bn/FusedBatchNormV3/ReadVariableOp_1(^model/block1b_project_bn/ReadVariableOp*^model/block1b_project_bn/ReadVariableOp_11^model/block1b_project_conv/Conv2D/ReadVariableOp/^model/block1b_se_expand/BiasAdd/ReadVariableOp.^model/block1b_se_expand/Conv2D/ReadVariableOp/^model/block1b_se_reduce/BiasAdd/ReadVariableOp.^model/block1b_se_reduce/Conv2D/ReadVariableOp$^model/conv2d/BiasAdd/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp%^model/dense_3/BiasAdd/ReadVariableOp$^model/dense_3/MatMul/ReadVariableOp+^model/normalization/Reshape/ReadVariableOp-^model/normalization/Reshape_1/ReadVariableOp%^model/output1/BiasAdd/ReadVariableOp$^model/output1/MatMul/ReadVariableOp%^model/output2/BiasAdd/ReadVariableOp$^model/output2/MatMul/ReadVariableOp%^model/output3/BiasAdd/ReadVariableOp$^model/output3/MatMul/ReadVariableOp%^model/output4/BiasAdd/ReadVariableOp$^model/output4/MatMul/ReadVariableOp.^model/stem_bn/FusedBatchNormV3/ReadVariableOp0^model/stem_bn/FusedBatchNormV3/ReadVariableOp_1^model/stem_bn/ReadVariableOp^model/stem_bn/ReadVariableOp_1&^model/stem_conv/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*
_input_shapesô
ñ:ÿÿÿÿÿÿÿÿÿ¬¬:::::::::::::::::::::::::::::::::::::::::::::::::::::2d
0model/block1a_bn/FusedBatchNormV3/ReadVariableOp0model/block1a_bn/FusedBatchNormV3/ReadVariableOp2h
2model/block1a_bn/FusedBatchNormV3/ReadVariableOp_12model/block1a_bn/FusedBatchNormV3/ReadVariableOp_12B
model/block1a_bn/ReadVariableOpmodel/block1a_bn/ReadVariableOp2F
!model/block1a_bn/ReadVariableOp_1!model/block1a_bn/ReadVariableOp_12^
-model/block1a_dwconv/depthwise/ReadVariableOp-model/block1a_dwconv/depthwise/ReadVariableOp2t
8model/block1a_project_bn/FusedBatchNormV3/ReadVariableOp8model/block1a_project_bn/FusedBatchNormV3/ReadVariableOp2x
:model/block1a_project_bn/FusedBatchNormV3/ReadVariableOp_1:model/block1a_project_bn/FusedBatchNormV3/ReadVariableOp_12R
'model/block1a_project_bn/ReadVariableOp'model/block1a_project_bn/ReadVariableOp2V
)model/block1a_project_bn/ReadVariableOp_1)model/block1a_project_bn/ReadVariableOp_12d
0model/block1a_project_conv/Conv2D/ReadVariableOp0model/block1a_project_conv/Conv2D/ReadVariableOp2`
.model/block1a_se_expand/BiasAdd/ReadVariableOp.model/block1a_se_expand/BiasAdd/ReadVariableOp2^
-model/block1a_se_expand/Conv2D/ReadVariableOp-model/block1a_se_expand/Conv2D/ReadVariableOp2`
.model/block1a_se_reduce/BiasAdd/ReadVariableOp.model/block1a_se_reduce/BiasAdd/ReadVariableOp2^
-model/block1a_se_reduce/Conv2D/ReadVariableOp-model/block1a_se_reduce/Conv2D/ReadVariableOp2d
0model/block1b_bn/FusedBatchNormV3/ReadVariableOp0model/block1b_bn/FusedBatchNormV3/ReadVariableOp2h
2model/block1b_bn/FusedBatchNormV3/ReadVariableOp_12model/block1b_bn/FusedBatchNormV3/ReadVariableOp_12B
model/block1b_bn/ReadVariableOpmodel/block1b_bn/ReadVariableOp2F
!model/block1b_bn/ReadVariableOp_1!model/block1b_bn/ReadVariableOp_12^
-model/block1b_dwconv/depthwise/ReadVariableOp-model/block1b_dwconv/depthwise/ReadVariableOp2t
8model/block1b_project_bn/FusedBatchNormV3/ReadVariableOp8model/block1b_project_bn/FusedBatchNormV3/ReadVariableOp2x
:model/block1b_project_bn/FusedBatchNormV3/ReadVariableOp_1:model/block1b_project_bn/FusedBatchNormV3/ReadVariableOp_12R
'model/block1b_project_bn/ReadVariableOp'model/block1b_project_bn/ReadVariableOp2V
)model/block1b_project_bn/ReadVariableOp_1)model/block1b_project_bn/ReadVariableOp_12d
0model/block1b_project_conv/Conv2D/ReadVariableOp0model/block1b_project_conv/Conv2D/ReadVariableOp2`
.model/block1b_se_expand/BiasAdd/ReadVariableOp.model/block1b_se_expand/BiasAdd/ReadVariableOp2^
-model/block1b_se_expand/Conv2D/ReadVariableOp-model/block1b_se_expand/Conv2D/ReadVariableOp2`
.model/block1b_se_reduce/BiasAdd/ReadVariableOp.model/block1b_se_reduce/BiasAdd/ReadVariableOp2^
-model/block1b_se_reduce/Conv2D/ReadVariableOp-model/block1b_se_reduce/Conv2D/ReadVariableOp2J
#model/conv2d/BiasAdd/ReadVariableOp#model/conv2d/BiasAdd/ReadVariableOp2H
"model/conv2d/Conv2D/ReadVariableOp"model/conv2d/Conv2D/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2L
$model/dense_2/BiasAdd/ReadVariableOp$model/dense_2/BiasAdd/ReadVariableOp2J
#model/dense_2/MatMul/ReadVariableOp#model/dense_2/MatMul/ReadVariableOp2L
$model/dense_3/BiasAdd/ReadVariableOp$model/dense_3/BiasAdd/ReadVariableOp2J
#model/dense_3/MatMul/ReadVariableOp#model/dense_3/MatMul/ReadVariableOp2X
*model/normalization/Reshape/ReadVariableOp*model/normalization/Reshape/ReadVariableOp2\
,model/normalization/Reshape_1/ReadVariableOp,model/normalization/Reshape_1/ReadVariableOp2L
$model/output1/BiasAdd/ReadVariableOp$model/output1/BiasAdd/ReadVariableOp2J
#model/output1/MatMul/ReadVariableOp#model/output1/MatMul/ReadVariableOp2L
$model/output2/BiasAdd/ReadVariableOp$model/output2/BiasAdd/ReadVariableOp2J
#model/output2/MatMul/ReadVariableOp#model/output2/MatMul/ReadVariableOp2L
$model/output3/BiasAdd/ReadVariableOp$model/output3/BiasAdd/ReadVariableOp2J
#model/output3/MatMul/ReadVariableOp#model/output3/MatMul/ReadVariableOp2L
$model/output4/BiasAdd/ReadVariableOp$model/output4/BiasAdd/ReadVariableOp2J
#model/output4/MatMul/ReadVariableOp#model/output4/MatMul/ReadVariableOp2^
-model/stem_bn/FusedBatchNormV3/ReadVariableOp-model/stem_bn/FusedBatchNormV3/ReadVariableOp2b
/model/stem_bn/FusedBatchNormV3/ReadVariableOp_1/model/stem_bn/FusedBatchNormV3/ReadVariableOp_12<
model/stem_bn/ReadVariableOpmodel/stem_bn/ReadVariableOp2@
model/stem_bn/ReadVariableOp_1model/stem_bn/ReadVariableOp_12N
%model/stem_conv/Conv2D/ReadVariableOp%model/stem_conv/Conv2D/ReadVariableOp:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬
!
_user_specified_name	input_1
²
å
B__inference_stem_bn_layer_call_and_return_conditional_losses_98068

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:(*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:(*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:(*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:(*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(:(:(:(:(:*
epsilon%o:*
is_training( 2
FusedBatchNormV3ì
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
é
y
M__inference_block1a_se_excite_layer_call_and_return_conditional_losses_101582
inputs_0
inputs_1
identitya
mulMulinputs_0inputs_1*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
mule
IdentityIdentitymul:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
"
_user_specified_name
inputs/1
Í
O
3__inference_block1a_activation_layer_call_fn_101512

inputs
identityÕ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_block1a_activation_layer_call_and_return_conditional_losses_988102
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
Ö
«
P__inference_block1b_project_conv_layer_call_and_return_conditional_losses_101948

inputs"
conv2d_readvariableop_resource
identity¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ç
L
0__inference_stem_activation_layer_call_fn_101373

inputs
identityÒ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_stem_activation_layer_call_and_return_conditional_losses_987182
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
þ
ñ
N__inference_block1b_project_bn_layer_call_and_return_conditional_losses_102035

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ì
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ü
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ý
ð
M__inference_block1a_project_bn_layer_call_and_return_conditional_losses_98969

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ì
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ü
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
é
s
G__inference_block1b_add_layer_call_and_return_conditional_losses_102120
inputs_0
inputs_1
identityc
addAddV2inputs_0inputs_1*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
adde
IdentityIdentityadd:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*M
_input_shapes<
::ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
	
Ü
C__inference_output4_layer_call_and_return_conditional_losses_102303

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø
X
,__inference_block1b_add_layer_call_fn_102126
inputs_0
inputs_1
identityÛ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block1b_add_layer_call_and_return_conditional_losses_993282
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*M
_input_shapes<
::ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
å
p
*__inference_stem_conv_layer_call_fn_101234

inputs
unknown
identity¢StatefulPartitionedCallñ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_stem_conv_layer_call_and_return_conditional_losses_986252
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ­­:22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­­
 
_user_specified_nameinputs
Í
O
3__inference_block1b_activation_layer_call_fn_101865

inputs
identityÕ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_block1b_activation_layer_call_and_return_conditional_losses_990892
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
è
E__inference_block1b_bn_layer_call_and_return_conditional_losses_98448

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3ì
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

k
M__inference_block1a_activation_layer_call_and_return_conditional_losses_98810

inputs

identity_1a
SigmoidSigmoidinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2	
Sigmoidb
mulMulinputsSigmoid:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
mule
IdentityIdentitymul:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity¾
	IdentityN	IdentityNmul:z:0inputs*
T
2*+
_gradient_op_typeCustomGradient-98803*N
_output_shapes<
::ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(2
	IdentityNt

Identity_1IdentityIdentityN:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity_1"!

identity_1Identity_1:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
Í
f
-__inference_block1b_drop_layer_call_fn_102109

inputs
identity¢StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_block1b_drop_layer_call_and_return_conditional_losses_993042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
þ
ñ
N__inference_block1a_project_bn_layer_call_and_return_conditional_losses_101682

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ì
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ü
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿ
h
J__inference_stem_activation_layer_call_and_return_conditional_losses_98718

inputs

identity_1a
SigmoidSigmoidinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2	
Sigmoidb
mulMulinputsSigmoid:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
mule
IdentityIdentitymul:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity¾
	IdentityN	IdentityNmul:z:0inputs*
T
2*+
_gradient_op_typeCustomGradient-98711*N
_output_shapes<
::ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(2
	IdentityNt

Identity_1IdentityIdentityN:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity_1"!

identity_1Identity_1:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
¶
Ò
&__inference_model_layer_call_fn_101220

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51
identity

identity_1

identity_2

identity_3¢StatefulPartitionedCallî
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51*A
Tin:
826*
Tout
2*
_collective_manager_ids
 *`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*W
_read_only_resource_inputs9
75	
 !"#$%&'()*+,-./012345*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_1002072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*
_input_shapesô
ñ:ÿÿÿÿÿÿÿÿÿ¬¬:::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬
 
_user_specified_nameinputs

¦
3__inference_block1b_project_bn_layer_call_fn_102004

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall±
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_block1b_project_bn_layer_call_and_return_conditional_losses_985302
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â

(__inference_stem_bn_layer_call_fn_101345

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_stem_bn_layer_call_and_return_conditional_losses_986542
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ(::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
©
f
G__inference_block1b_drop_layer_call_and_return_conditional_losses_99304

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceV
packed/1Const*
_output_shapes
: *
dtype0*
value	B :2

packed/1V
packed/2Const*
_output_shapes
: *
dtype0*
value	B :2

packed/2V
packed/3Const*
_output_shapes
: *
dtype0*
value	B :2

packed/3
packedPackstrided_slice:output:0packed/1:output:0packed/2:output:0packed/3:output:0*
N*
T0*
_output_shapes
:2
packedc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *þ?2
dropout/Const}
dropout/MulMulinputsdropout/Const:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mulµ
$dropout/random_uniform/RandomUniformRandomUniformpacked:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Áü;2
dropout/GreaterEqual/yÆ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1o
IdentityIdentitydropout/Mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
é
y
M__inference_block1b_se_excite_layer_call_and_return_conditional_losses_101935
inputs_0
inputs_1
identitya
mulMulinputs_0inputs_1*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mule
IdentityIdentitymul:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
õ
è
E__inference_block1b_bn_layer_call_and_return_conditional_losses_99025

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ì
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ü
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ò
å
B__inference_stem_bn_layer_call_and_return_conditional_losses_98672

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:(*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:(*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:(*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:(*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ì
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ(:(:(:(:(:*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ü
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ(::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
	
Ü
C__inference_output1_layer_call_and_return_conditional_losses_102246

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
û
{
5__inference_block1a_project_conv_layer_call_fn_101602

inputs
unknown
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_block1a_project_conv_layer_call_and_return_conditional_losses_989222
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ(:22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
è	
Ú
A__inference_conv2d_layer_call_and_return_conditional_losses_99348

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â

(__inference_stem_bn_layer_call_fn_101358

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_stem_bn_layer_call_and_return_conditional_losses_986722
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ(::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
â
d
H__inference_stem_conv_pad_layer_call_and_return_conditional_losses_98004

inputs
identity
Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                               2
Pad/paddings
PadPadinputsPad/paddings:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Pad
IdentityIdentityPad:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø
¦
3__inference_block1a_project_bn_layer_call_fn_101726

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¡
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_block1a_project_bn_layer_call_and_return_conditional_losses_989692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¦
3__inference_block1a_project_bn_layer_call_fn_101664

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall±
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_block1a_project_bn_layer_call_and_return_conditional_losses_983302
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¯
M
1__inference_average_pooling2d_layer_call_fn_98584

inputs
identityí
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_985782
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ï
j
N__inference_block1b_se_reshape_layer_call_and_return_conditional_losses_101879

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3º
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ð	
Ü
C__inference_dense_3_layer_call_and_return_conditional_losses_102227

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


+__inference_block1a_bn_layer_call_fn_101422

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_block1a_bn_layer_call_and_return_conditional_losses_981862
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
½
ð
M__inference_block1b_project_bn_layer_call_and_return_conditional_losses_98561

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3ì
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª
g
H__inference_block1b_drop_layer_call_and_return_conditional_losses_102099

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceV
packed/1Const*
_output_shapes
: *
dtype0*
value	B :2

packed/1V
packed/2Const*
_output_shapes
: *
dtype0*
value	B :2

packed/2V
packed/3Const*
_output_shapes
: *
dtype0*
value	B :2

packed/3
packedPackstrided_slice:output:0packed/1:output:0packed/2:output:0packed/3:output:0*
N*
T0*
_output_shapes
:2
packedc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *þ?2
dropout/Const}
dropout/MulMulinputsdropout/Const:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mulµ
$dropout/random_uniform/RandomUniformRandomUniformpacked:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Áü;2
dropout/GreaterEqual/yÆ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1o
IdentityIdentitydropout/Mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
à
v
L__inference_block1a_se_excite_layer_call_and_return_conditional_losses_98906

inputs
inputs_1
identity_
mulMulinputsinputs_1*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
mule
IdentityIdentitymul:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs:WS
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
õ
è
E__inference_block1a_bn_layer_call_and_return_conditional_losses_98764

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:(*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:(*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:(*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:(*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ì
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ(:(:(:(:(:*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ü
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ(::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
ö
é
F__inference_block1a_bn_layer_call_and_return_conditional_losses_101453

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:(*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:(*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:(*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:(*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ì
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ(:(:(:(:(:*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ü
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ(::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs

¦
3__inference_block1b_project_bn_layer_call_fn_102017

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall±
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_block1b_project_bn_layer_call_and_return_conditional_losses_985612
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ý
ð
M__inference_block1a_project_bn_layer_call_and_return_conditional_losses_98951

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ì
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ü
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®
t
.__inference_block1b_dwconv_layer_call_fn_98359

inputs
unknown
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_block1b_dwconv_layer_call_and_return_conditional_losses_983512
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

l
N__inference_block1b_activation_layer_call_and_return_conditional_losses_101860

inputs

identity_1a
SigmoidSigmoidinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidb
mulMulinputsSigmoid:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mule
IdentityIdentitymul:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¿
	IdentityN	IdentityNmul:z:0inputs*
T
2*,
_gradient_op_typeCustomGradient-101853*N
_output_shapes<
::ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ2
	IdentityNt

Identity_1IdentityIdentityN:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»
_
C__inference_flatten_layer_call_and_return_conditional_losses_102151

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
ñ
N__inference_block1b_project_bn_layer_call_and_return_conditional_losses_101991

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3ì
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¸
Ó
&__inference_model_layer_call_fn_100041
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51
identity

identity_1

identity_2

identity_3¢StatefulPartitionedCallî
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51*A
Tin:
826*
Tout
2*
_collective_manager_ids
 *`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*W
_read_only_resource_inputs9
75	
 !"#$%&'()*+,-./012345*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_999262
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*
_input_shapesô
ñ:ÿÿÿÿÿÿÿÿÿ¬¬:::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬
!
_user_specified_name	input_1
ï	
Û
B__inference_dense_2_layer_call_and_return_conditional_losses_99417

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È

+__inference_block1a_bn_layer_call_fn_101497

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_block1a_bn_layer_call_and_return_conditional_losses_987642
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ(::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
¾
ñ
N__inference_block1a_project_bn_layer_call_and_return_conditional_losses_101620

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3ì
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ç
L__inference_block1b_se_reduce_layer_call_and_return_conditional_losses_99136

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource

identity_1¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddi
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidj
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulc
IdentityIdentitymul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÄ
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*+
_gradient_op_typeCustomGradient-99129*J
_output_shapes8
6:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ2
	IdentityN£

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Àß
î
@__inference_model_layer_call_and_return_conditional_losses_99759
input_11
-normalization_reshape_readvariableop_resource3
/normalization_reshape_1_readvariableop_resource
stem_conv_99616
stem_bn_99619
stem_bn_99621
stem_bn_99623
stem_bn_99625
block1a_dwconv_99629
block1a_bn_99632
block1a_bn_99634
block1a_bn_99636
block1a_bn_99638
block1a_se_reduce_99644
block1a_se_reduce_99646
block1a_se_expand_99649
block1a_se_expand_99651
block1a_project_conv_99655
block1a_project_bn_99658
block1a_project_bn_99660
block1a_project_bn_99662
block1a_project_bn_99664
block1b_dwconv_99667
block1b_bn_99670
block1b_bn_99672
block1b_bn_99674
block1b_bn_99676
block1b_se_reduce_99682
block1b_se_reduce_99684
block1b_se_expand_99687
block1b_se_expand_99689
block1b_project_conv_99693
block1b_project_bn_99696
block1b_project_bn_99698
block1b_project_bn_99700
block1b_project_bn_99702
conv2d_99708
conv2d_99710
dense_3_99715
dense_3_99717
dense_2_99720
dense_2_99722
dense_1_99725
dense_1_99727
dense_99730
dense_99732
output4_99735
output4_99737
output3_99740
output3_99742
output2_99745
output2_99747
output1_99750
output1_99752
identity

identity_1

identity_2

identity_3¢"block1a_bn/StatefulPartitionedCall¢&block1a_dwconv/StatefulPartitionedCall¢*block1a_project_bn/StatefulPartitionedCall¢,block1a_project_conv/StatefulPartitionedCall¢)block1a_se_expand/StatefulPartitionedCall¢)block1a_se_reduce/StatefulPartitionedCall¢"block1b_bn/StatefulPartitionedCall¢&block1b_dwconv/StatefulPartitionedCall¢*block1b_project_bn/StatefulPartitionedCall¢,block1b_project_conv/StatefulPartitionedCall¢)block1b_se_expand/StatefulPartitionedCall¢)block1b_se_reduce/StatefulPartitionedCall¢conv2d/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢$normalization/Reshape/ReadVariableOp¢&normalization/Reshape_1/ReadVariableOp¢output1/StatefulPartitionedCall¢output2/StatefulPartitionedCall¢output3/StatefulPartitionedCall¢output4/StatefulPartitionedCall¢stem_bn/StatefulPartitionedCall¢!stem_conv/StatefulPartitionedCalli
rescaling/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;2
rescaling/Cast/xm
rescaling/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rescaling/Cast_1/x
rescaling/mulMulinput_1rescaling/Cast/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬2
rescaling/mul
rescaling/addAddV2rescaling/mul:z:0rescaling/Cast_1/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬2
rescaling/add¶
$normalization/Reshape/ReadVariableOpReadVariableOp-normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02&
$normalization/Reshape/ReadVariableOp
normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
normalization/Reshape/shape¾
normalization/ReshapeReshape,normalization/Reshape/ReadVariableOp:value:0$normalization/Reshape/shape:output:0*
T0*&
_output_shapes
:2
normalization/Reshape¼
&normalization/Reshape_1/ReadVariableOpReadVariableOp/normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization/Reshape_1/ReadVariableOp
normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
normalization/Reshape_1/shapeÆ
normalization/Reshape_1Reshape.normalization/Reshape_1/ReadVariableOp:value:0&normalization/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
normalization/Reshape_1
normalization/subSubrescaling/add:z:0normalization/Reshape:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬2
normalization/sub
normalization/SqrtSqrt normalization/Reshape_1:output:0*
T0*&
_output_shapes
:2
normalization/Sqrtw
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö32
normalization/Maximum/y¤
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*&
_output_shapes
:2
normalization/Maximum§
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬2
normalization/truedivÿ
stem_conv_pad/PartitionedCallPartitionedCallnormalization/truediv:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­­* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_stem_conv_pad_layer_call_and_return_conditional_losses_980042
stem_conv_pad/PartitionedCall­
!stem_conv/StatefulPartitionedCallStatefulPartitionedCall&stem_conv_pad/PartitionedCall:output:0stem_conv_99616*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_stem_conv_layer_call_and_return_conditional_losses_986252#
!stem_conv/StatefulPartitionedCallÜ
stem_bn/StatefulPartitionedCallStatefulPartitionedCall*stem_conv/StatefulPartitionedCall:output:0stem_bn_99619stem_bn_99621stem_bn_99623stem_bn_99625*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_stem_bn_layer_call_and_return_conditional_losses_986722!
stem_bn/StatefulPartitionedCall
stem_activation/PartitionedCallPartitionedCall(stem_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_stem_activation_layer_call_and_return_conditional_losses_987182!
stem_activation/PartitionedCallÃ
&block1a_dwconv/StatefulPartitionedCallStatefulPartitionedCall(stem_activation/PartitionedCall:output:0block1a_dwconv_99629*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_block1a_dwconv_layer_call_and_return_conditional_losses_981202(
&block1a_dwconv/StatefulPartitionedCallö
"block1a_bn/StatefulPartitionedCallStatefulPartitionedCall/block1a_dwconv/StatefulPartitionedCall:output:0block1a_bn_99632block1a_bn_99634block1a_bn_99636block1a_bn_99638*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_block1a_bn_layer_call_and_return_conditional_losses_987642$
"block1a_bn/StatefulPartitionedCall 
"block1a_activation/PartitionedCallPartitionedCall+block1a_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_block1a_activation_layer_call_and_return_conditional_losses_988102$
"block1a_activation/PartitionedCall
"block1a_se_squeeze/PartitionedCallPartitionedCall+block1a_activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_block1a_se_squeeze_layer_call_and_return_conditional_losses_982352$
"block1a_se_squeeze/PartitionedCall
"block1a_se_reshape/PartitionedCallPartitionedCall+block1a_se_squeeze/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_block1a_se_reshape_layer_call_and_return_conditional_losses_988332$
"block1a_se_reshape/PartitionedCallë
)block1a_se_reduce/StatefulPartitionedCallStatefulPartitionedCall+block1a_se_reshape/PartitionedCall:output:0block1a_se_reduce_99644block1a_se_reduce_99646*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_block1a_se_reduce_layer_call_and_return_conditional_losses_988572+
)block1a_se_reduce/StatefulPartitionedCallò
)block1a_se_expand/StatefulPartitionedCallStatefulPartitionedCall2block1a_se_reduce/StatefulPartitionedCall:output:0block1a_se_expand_99649block1a_se_expand_99651*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_block1a_se_expand_layer_call_and_return_conditional_losses_988842+
)block1a_se_expand/StatefulPartitionedCallÒ
!block1a_se_excite/PartitionedCallPartitionedCall+block1a_activation/PartitionedCall:output:02block1a_se_expand/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_block1a_se_excite_layer_call_and_return_conditional_losses_989062#
!block1a_se_excite/PartitionedCallÝ
,block1a_project_conv/StatefulPartitionedCallStatefulPartitionedCall*block1a_se_excite/PartitionedCall:output:0block1a_project_conv_99655*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_block1a_project_conv_layer_call_and_return_conditional_losses_989222.
,block1a_project_conv/StatefulPartitionedCall´
*block1a_project_bn/StatefulPartitionedCallStatefulPartitionedCall5block1a_project_conv/StatefulPartitionedCall:output:0block1a_project_bn_99658block1a_project_bn_99660block1a_project_bn_99662block1a_project_bn_99664*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_block1a_project_bn_layer_call_and_return_conditional_losses_989692,
*block1a_project_bn/StatefulPartitionedCallÎ
&block1b_dwconv/StatefulPartitionedCallStatefulPartitionedCall3block1a_project_bn/StatefulPartitionedCall:output:0block1b_dwconv_99667*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_block1b_dwconv_layer_call_and_return_conditional_losses_983512(
&block1b_dwconv/StatefulPartitionedCallö
"block1b_bn/StatefulPartitionedCallStatefulPartitionedCall/block1b_dwconv/StatefulPartitionedCall:output:0block1b_bn_99670block1b_bn_99672block1b_bn_99674block1b_bn_99676*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_block1b_bn_layer_call_and_return_conditional_losses_990432$
"block1b_bn/StatefulPartitionedCall 
"block1b_activation/PartitionedCallPartitionedCall+block1b_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_block1b_activation_layer_call_and_return_conditional_losses_990892$
"block1b_activation/PartitionedCall
"block1b_se_squeeze/PartitionedCallPartitionedCall+block1b_activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_block1b_se_squeeze_layer_call_and_return_conditional_losses_984662$
"block1b_se_squeeze/PartitionedCall
"block1b_se_reshape/PartitionedCallPartitionedCall+block1b_se_squeeze/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_block1b_se_reshape_layer_call_and_return_conditional_losses_991122$
"block1b_se_reshape/PartitionedCallë
)block1b_se_reduce/StatefulPartitionedCallStatefulPartitionedCall+block1b_se_reshape/PartitionedCall:output:0block1b_se_reduce_99682block1b_se_reduce_99684*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_block1b_se_reduce_layer_call_and_return_conditional_losses_991362+
)block1b_se_reduce/StatefulPartitionedCallò
)block1b_se_expand/StatefulPartitionedCallStatefulPartitionedCall2block1b_se_reduce/StatefulPartitionedCall:output:0block1b_se_expand_99687block1b_se_expand_99689*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_block1b_se_expand_layer_call_and_return_conditional_losses_991632+
)block1b_se_expand/StatefulPartitionedCallÒ
!block1b_se_excite/PartitionedCallPartitionedCall+block1b_activation/PartitionedCall:output:02block1b_se_expand/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_block1b_se_excite_layer_call_and_return_conditional_losses_991852#
!block1b_se_excite/PartitionedCallÝ
,block1b_project_conv/StatefulPartitionedCallStatefulPartitionedCall*block1b_se_excite/PartitionedCall:output:0block1b_project_conv_99693*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_block1b_project_conv_layer_call_and_return_conditional_losses_992012.
,block1b_project_conv/StatefulPartitionedCall´
*block1b_project_bn/StatefulPartitionedCallStatefulPartitionedCall5block1b_project_conv/StatefulPartitionedCall:output:0block1b_project_bn_99696block1b_project_bn_99698block1b_project_bn_99700block1b_project_bn_99702*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_block1b_project_bn_layer_call_and_return_conditional_losses_992482,
*block1b_project_bn/StatefulPartitionedCall
block1b_drop/PartitionedCallPartitionedCall3block1b_project_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_block1b_drop_layer_call_and_return_conditional_losses_993092
block1b_drop/PartitionedCall»
block1b_add/PartitionedCallPartitionedCall%block1b_drop/PartitionedCall:output:03block1a_project_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block1b_add_layer_call_and_return_conditional_losses_993282
block1b_add/PartitionedCall
!average_pooling2d/PartitionedCallPartitionedCall$block1b_add/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_985782#
!average_pooling2d/PartitionedCall³
conv2d/StatefulPartitionedCallStatefulPartitionedCall*average_pooling2d/PartitionedCall:output:0conv2d_99708conv2d_99710*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_993482 
conv2d/StatefulPartitionedCall
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_985902
max_pooling2d/PartitionedCallñ
flatten/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_993712
flatten/PartitionedCall¦
dense_3/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_3_99715dense_3_99717*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_993902!
dense_3/StatefulPartitionedCall¦
dense_2/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_2_99720dense_2_99722*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_994172!
dense_2/StatefulPartitionedCall¦
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_99725dense_1_99727*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_994442!
dense_1/StatefulPartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_99730dense_99732*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_994712
dense/StatefulPartitionedCall®
output4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0output4_99735output4_99737*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_output4_layer_call_and_return_conditional_losses_994972!
output4/StatefulPartitionedCall®
output3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0output3_99740output3_99742*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_output3_layer_call_and_return_conditional_losses_995232!
output3/StatefulPartitionedCall®
output2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0output2_99745output2_99747*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_output2_layer_call_and_return_conditional_losses_995492!
output2/StatefulPartitionedCall¬
output1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0output1_99750output1_99752*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_output1_layer_call_and_return_conditional_losses_995752!
output1/StatefulPartitionedCallÅ
IdentityIdentity(output1/StatefulPartitionedCall:output:0#^block1a_bn/StatefulPartitionedCall'^block1a_dwconv/StatefulPartitionedCall+^block1a_project_bn/StatefulPartitionedCall-^block1a_project_conv/StatefulPartitionedCall*^block1a_se_expand/StatefulPartitionedCall*^block1a_se_reduce/StatefulPartitionedCall#^block1b_bn/StatefulPartitionedCall'^block1b_dwconv/StatefulPartitionedCall+^block1b_project_bn/StatefulPartitionedCall-^block1b_project_conv/StatefulPartitionedCall*^block1b_se_expand/StatefulPartitionedCall*^block1b_se_reduce/StatefulPartitionedCall^conv2d/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp ^output1/StatefulPartitionedCall ^output2/StatefulPartitionedCall ^output3/StatefulPartitionedCall ^output4/StatefulPartitionedCall ^stem_bn/StatefulPartitionedCall"^stem_conv/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÉ

Identity_1Identity(output2/StatefulPartitionedCall:output:0#^block1a_bn/StatefulPartitionedCall'^block1a_dwconv/StatefulPartitionedCall+^block1a_project_bn/StatefulPartitionedCall-^block1a_project_conv/StatefulPartitionedCall*^block1a_se_expand/StatefulPartitionedCall*^block1a_se_reduce/StatefulPartitionedCall#^block1b_bn/StatefulPartitionedCall'^block1b_dwconv/StatefulPartitionedCall+^block1b_project_bn/StatefulPartitionedCall-^block1b_project_conv/StatefulPartitionedCall*^block1b_se_expand/StatefulPartitionedCall*^block1b_se_reduce/StatefulPartitionedCall^conv2d/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp ^output1/StatefulPartitionedCall ^output2/StatefulPartitionedCall ^output3/StatefulPartitionedCall ^output4/StatefulPartitionedCall ^stem_bn/StatefulPartitionedCall"^stem_conv/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1É

Identity_2Identity(output3/StatefulPartitionedCall:output:0#^block1a_bn/StatefulPartitionedCall'^block1a_dwconv/StatefulPartitionedCall+^block1a_project_bn/StatefulPartitionedCall-^block1a_project_conv/StatefulPartitionedCall*^block1a_se_expand/StatefulPartitionedCall*^block1a_se_reduce/StatefulPartitionedCall#^block1b_bn/StatefulPartitionedCall'^block1b_dwconv/StatefulPartitionedCall+^block1b_project_bn/StatefulPartitionedCall-^block1b_project_conv/StatefulPartitionedCall*^block1b_se_expand/StatefulPartitionedCall*^block1b_se_reduce/StatefulPartitionedCall^conv2d/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp ^output1/StatefulPartitionedCall ^output2/StatefulPartitionedCall ^output3/StatefulPartitionedCall ^output4/StatefulPartitionedCall ^stem_bn/StatefulPartitionedCall"^stem_conv/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2É

Identity_3Identity(output4/StatefulPartitionedCall:output:0#^block1a_bn/StatefulPartitionedCall'^block1a_dwconv/StatefulPartitionedCall+^block1a_project_bn/StatefulPartitionedCall-^block1a_project_conv/StatefulPartitionedCall*^block1a_se_expand/StatefulPartitionedCall*^block1a_se_reduce/StatefulPartitionedCall#^block1b_bn/StatefulPartitionedCall'^block1b_dwconv/StatefulPartitionedCall+^block1b_project_bn/StatefulPartitionedCall-^block1b_project_conv/StatefulPartitionedCall*^block1b_se_expand/StatefulPartitionedCall*^block1b_se_reduce/StatefulPartitionedCall^conv2d/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp ^output1/StatefulPartitionedCall ^output2/StatefulPartitionedCall ^output3/StatefulPartitionedCall ^output4/StatefulPartitionedCall ^stem_bn/StatefulPartitionedCall"^stem_conv/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*
_input_shapesô
ñ:ÿÿÿÿÿÿÿÿÿ¬¬:::::::::::::::::::::::::::::::::::::::::::::::::::::2H
"block1a_bn/StatefulPartitionedCall"block1a_bn/StatefulPartitionedCall2P
&block1a_dwconv/StatefulPartitionedCall&block1a_dwconv/StatefulPartitionedCall2X
*block1a_project_bn/StatefulPartitionedCall*block1a_project_bn/StatefulPartitionedCall2\
,block1a_project_conv/StatefulPartitionedCall,block1a_project_conv/StatefulPartitionedCall2V
)block1a_se_expand/StatefulPartitionedCall)block1a_se_expand/StatefulPartitionedCall2V
)block1a_se_reduce/StatefulPartitionedCall)block1a_se_reduce/StatefulPartitionedCall2H
"block1b_bn/StatefulPartitionedCall"block1b_bn/StatefulPartitionedCall2P
&block1b_dwconv/StatefulPartitionedCall&block1b_dwconv/StatefulPartitionedCall2X
*block1b_project_bn/StatefulPartitionedCall*block1b_project_bn/StatefulPartitionedCall2\
,block1b_project_conv/StatefulPartitionedCall,block1b_project_conv/StatefulPartitionedCall2V
)block1b_se_expand/StatefulPartitionedCall)block1b_se_expand/StatefulPartitionedCall2V
)block1b_se_reduce/StatefulPartitionedCall)block1b_se_reduce/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2L
$normalization/Reshape/ReadVariableOp$normalization/Reshape/ReadVariableOp2P
&normalization/Reshape_1/ReadVariableOp&normalization/Reshape_1/ReadVariableOp2B
output1/StatefulPartitionedCalloutput1/StatefulPartitionedCall2B
output2/StatefulPartitionedCalloutput2/StatefulPartitionedCall2B
output3/StatefulPartitionedCalloutput3/StatefulPartitionedCall2B
output4/StatefulPartitionedCalloutput4/StatefulPartitionedCall2B
stem_bn/StatefulPartitionedCallstem_bn/StatefulPartitionedCall2F
!stem_conv/StatefulPartitionedCall!stem_conv/StatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬
!
_user_specified_name	input_1
ó
æ
C__inference_stem_bn_layer_call_and_return_conditional_losses_101314

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:(*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:(*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:(*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:(*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ì
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ(:(:(:(:(:*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ü
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ(::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
Ù
}
(__inference_output2_layer_call_fn_102274

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_output2_layer_call_and_return_conditional_losses_995492
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
õ
è
E__inference_block1a_bn_layer_call_and_return_conditional_losses_98746

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:(*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:(*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:(*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:(*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ì
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ(:(:(:(:(:*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ü
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ(::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
ñá

@__inference_model_layer_call_and_return_conditional_losses_99595
input_11
-normalization_reshape_readvariableop_resource3
/normalization_reshape_1_readvariableop_resource
stem_conv_98634
stem_bn_98699
stem_bn_98701
stem_bn_98703
stem_bn_98705
block1a_dwconv_98726
block1a_bn_98791
block1a_bn_98793
block1a_bn_98795
block1a_bn_98797
block1a_se_reduce_98868
block1a_se_reduce_98870
block1a_se_expand_98895
block1a_se_expand_98897
block1a_project_conv_98931
block1a_project_bn_98996
block1a_project_bn_98998
block1a_project_bn_99000
block1a_project_bn_99002
block1b_dwconv_99005
block1b_bn_99070
block1b_bn_99072
block1b_bn_99074
block1b_bn_99076
block1b_se_reduce_99147
block1b_se_reduce_99149
block1b_se_expand_99174
block1b_se_expand_99176
block1b_project_conv_99210
block1b_project_bn_99275
block1b_project_bn_99277
block1b_project_bn_99279
block1b_project_bn_99281
conv2d_99359
conv2d_99361
dense_3_99401
dense_3_99403
dense_2_99428
dense_2_99430
dense_1_99455
dense_1_99457
dense_99482
dense_99484
output4_99508
output4_99510
output3_99534
output3_99536
output2_99560
output2_99562
output1_99586
output1_99588
identity

identity_1

identity_2

identity_3¢"block1a_bn/StatefulPartitionedCall¢&block1a_dwconv/StatefulPartitionedCall¢*block1a_project_bn/StatefulPartitionedCall¢,block1a_project_conv/StatefulPartitionedCall¢)block1a_se_expand/StatefulPartitionedCall¢)block1a_se_reduce/StatefulPartitionedCall¢"block1b_bn/StatefulPartitionedCall¢$block1b_drop/StatefulPartitionedCall¢&block1b_dwconv/StatefulPartitionedCall¢*block1b_project_bn/StatefulPartitionedCall¢,block1b_project_conv/StatefulPartitionedCall¢)block1b_se_expand/StatefulPartitionedCall¢)block1b_se_reduce/StatefulPartitionedCall¢conv2d/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢$normalization/Reshape/ReadVariableOp¢&normalization/Reshape_1/ReadVariableOp¢output1/StatefulPartitionedCall¢output2/StatefulPartitionedCall¢output3/StatefulPartitionedCall¢output4/StatefulPartitionedCall¢stem_bn/StatefulPartitionedCall¢!stem_conv/StatefulPartitionedCalli
rescaling/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;2
rescaling/Cast/xm
rescaling/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rescaling/Cast_1/x
rescaling/mulMulinput_1rescaling/Cast/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬2
rescaling/mul
rescaling/addAddV2rescaling/mul:z:0rescaling/Cast_1/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬2
rescaling/add¶
$normalization/Reshape/ReadVariableOpReadVariableOp-normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02&
$normalization/Reshape/ReadVariableOp
normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
normalization/Reshape/shape¾
normalization/ReshapeReshape,normalization/Reshape/ReadVariableOp:value:0$normalization/Reshape/shape:output:0*
T0*&
_output_shapes
:2
normalization/Reshape¼
&normalization/Reshape_1/ReadVariableOpReadVariableOp/normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization/Reshape_1/ReadVariableOp
normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
normalization/Reshape_1/shapeÆ
normalization/Reshape_1Reshape.normalization/Reshape_1/ReadVariableOp:value:0&normalization/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
normalization/Reshape_1
normalization/subSubrescaling/add:z:0normalization/Reshape:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬2
normalization/sub
normalization/SqrtSqrt normalization/Reshape_1:output:0*
T0*&
_output_shapes
:2
normalization/Sqrtw
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö32
normalization/Maximum/y¤
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*&
_output_shapes
:2
normalization/Maximum§
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬2
normalization/truedivÿ
stem_conv_pad/PartitionedCallPartitionedCallnormalization/truediv:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­­* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_stem_conv_pad_layer_call_and_return_conditional_losses_980042
stem_conv_pad/PartitionedCall­
!stem_conv/StatefulPartitionedCallStatefulPartitionedCall&stem_conv_pad/PartitionedCall:output:0stem_conv_98634*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_stem_conv_layer_call_and_return_conditional_losses_986252#
!stem_conv/StatefulPartitionedCallÜ
stem_bn/StatefulPartitionedCallStatefulPartitionedCall*stem_conv/StatefulPartitionedCall:output:0stem_bn_98699stem_bn_98701stem_bn_98703stem_bn_98705*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_stem_bn_layer_call_and_return_conditional_losses_986542!
stem_bn/StatefulPartitionedCall
stem_activation/PartitionedCallPartitionedCall(stem_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_stem_activation_layer_call_and_return_conditional_losses_987182!
stem_activation/PartitionedCallÃ
&block1a_dwconv/StatefulPartitionedCallStatefulPartitionedCall(stem_activation/PartitionedCall:output:0block1a_dwconv_98726*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_block1a_dwconv_layer_call_and_return_conditional_losses_981202(
&block1a_dwconv/StatefulPartitionedCallö
"block1a_bn/StatefulPartitionedCallStatefulPartitionedCall/block1a_dwconv/StatefulPartitionedCall:output:0block1a_bn_98791block1a_bn_98793block1a_bn_98795block1a_bn_98797*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_block1a_bn_layer_call_and_return_conditional_losses_987462$
"block1a_bn/StatefulPartitionedCall 
"block1a_activation/PartitionedCallPartitionedCall+block1a_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_block1a_activation_layer_call_and_return_conditional_losses_988102$
"block1a_activation/PartitionedCall
"block1a_se_squeeze/PartitionedCallPartitionedCall+block1a_activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_block1a_se_squeeze_layer_call_and_return_conditional_losses_982352$
"block1a_se_squeeze/PartitionedCall
"block1a_se_reshape/PartitionedCallPartitionedCall+block1a_se_squeeze/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_block1a_se_reshape_layer_call_and_return_conditional_losses_988332$
"block1a_se_reshape/PartitionedCallë
)block1a_se_reduce/StatefulPartitionedCallStatefulPartitionedCall+block1a_se_reshape/PartitionedCall:output:0block1a_se_reduce_98868block1a_se_reduce_98870*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_block1a_se_reduce_layer_call_and_return_conditional_losses_988572+
)block1a_se_reduce/StatefulPartitionedCallò
)block1a_se_expand/StatefulPartitionedCallStatefulPartitionedCall2block1a_se_reduce/StatefulPartitionedCall:output:0block1a_se_expand_98895block1a_se_expand_98897*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_block1a_se_expand_layer_call_and_return_conditional_losses_988842+
)block1a_se_expand/StatefulPartitionedCallÒ
!block1a_se_excite/PartitionedCallPartitionedCall+block1a_activation/PartitionedCall:output:02block1a_se_expand/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_block1a_se_excite_layer_call_and_return_conditional_losses_989062#
!block1a_se_excite/PartitionedCallÝ
,block1a_project_conv/StatefulPartitionedCallStatefulPartitionedCall*block1a_se_excite/PartitionedCall:output:0block1a_project_conv_98931*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_block1a_project_conv_layer_call_and_return_conditional_losses_989222.
,block1a_project_conv/StatefulPartitionedCall´
*block1a_project_bn/StatefulPartitionedCallStatefulPartitionedCall5block1a_project_conv/StatefulPartitionedCall:output:0block1a_project_bn_98996block1a_project_bn_98998block1a_project_bn_99000block1a_project_bn_99002*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_block1a_project_bn_layer_call_and_return_conditional_losses_989512,
*block1a_project_bn/StatefulPartitionedCallÎ
&block1b_dwconv/StatefulPartitionedCallStatefulPartitionedCall3block1a_project_bn/StatefulPartitionedCall:output:0block1b_dwconv_99005*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_block1b_dwconv_layer_call_and_return_conditional_losses_983512(
&block1b_dwconv/StatefulPartitionedCallö
"block1b_bn/StatefulPartitionedCallStatefulPartitionedCall/block1b_dwconv/StatefulPartitionedCall:output:0block1b_bn_99070block1b_bn_99072block1b_bn_99074block1b_bn_99076*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_block1b_bn_layer_call_and_return_conditional_losses_990252$
"block1b_bn/StatefulPartitionedCall 
"block1b_activation/PartitionedCallPartitionedCall+block1b_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_block1b_activation_layer_call_and_return_conditional_losses_990892$
"block1b_activation/PartitionedCall
"block1b_se_squeeze/PartitionedCallPartitionedCall+block1b_activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_block1b_se_squeeze_layer_call_and_return_conditional_losses_984662$
"block1b_se_squeeze/PartitionedCall
"block1b_se_reshape/PartitionedCallPartitionedCall+block1b_se_squeeze/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_block1b_se_reshape_layer_call_and_return_conditional_losses_991122$
"block1b_se_reshape/PartitionedCallë
)block1b_se_reduce/StatefulPartitionedCallStatefulPartitionedCall+block1b_se_reshape/PartitionedCall:output:0block1b_se_reduce_99147block1b_se_reduce_99149*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_block1b_se_reduce_layer_call_and_return_conditional_losses_991362+
)block1b_se_reduce/StatefulPartitionedCallò
)block1b_se_expand/StatefulPartitionedCallStatefulPartitionedCall2block1b_se_reduce/StatefulPartitionedCall:output:0block1b_se_expand_99174block1b_se_expand_99176*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_block1b_se_expand_layer_call_and_return_conditional_losses_991632+
)block1b_se_expand/StatefulPartitionedCallÒ
!block1b_se_excite/PartitionedCallPartitionedCall+block1b_activation/PartitionedCall:output:02block1b_se_expand/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_block1b_se_excite_layer_call_and_return_conditional_losses_991852#
!block1b_se_excite/PartitionedCallÝ
,block1b_project_conv/StatefulPartitionedCallStatefulPartitionedCall*block1b_se_excite/PartitionedCall:output:0block1b_project_conv_99210*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_block1b_project_conv_layer_call_and_return_conditional_losses_992012.
,block1b_project_conv/StatefulPartitionedCall´
*block1b_project_bn/StatefulPartitionedCallStatefulPartitionedCall5block1b_project_conv/StatefulPartitionedCall:output:0block1b_project_bn_99275block1b_project_bn_99277block1b_project_bn_99279block1b_project_bn_99281*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_block1b_project_bn_layer_call_and_return_conditional_losses_992302,
*block1b_project_bn/StatefulPartitionedCall®
$block1b_drop/StatefulPartitionedCallStatefulPartitionedCall3block1b_project_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_block1b_drop_layer_call_and_return_conditional_losses_993042&
$block1b_drop/StatefulPartitionedCallÃ
block1b_add/PartitionedCallPartitionedCall-block1b_drop/StatefulPartitionedCall:output:03block1a_project_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block1b_add_layer_call_and_return_conditional_losses_993282
block1b_add/PartitionedCall
!average_pooling2d/PartitionedCallPartitionedCall$block1b_add/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_985782#
!average_pooling2d/PartitionedCall³
conv2d/StatefulPartitionedCallStatefulPartitionedCall*average_pooling2d/PartitionedCall:output:0conv2d_99359conv2d_99361*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_993482 
conv2d/StatefulPartitionedCall
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_985902
max_pooling2d/PartitionedCallñ
flatten/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_993712
flatten/PartitionedCall¦
dense_3/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_3_99401dense_3_99403*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_993902!
dense_3/StatefulPartitionedCall¦
dense_2/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_2_99428dense_2_99430*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_994172!
dense_2/StatefulPartitionedCall¦
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_99455dense_1_99457*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_994442!
dense_1/StatefulPartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_99482dense_99484*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_994712
dense/StatefulPartitionedCall®
output4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0output4_99508output4_99510*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_output4_layer_call_and_return_conditional_losses_994972!
output4/StatefulPartitionedCall®
output3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0output3_99534output3_99536*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_output3_layer_call_and_return_conditional_losses_995232!
output3/StatefulPartitionedCall®
output2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0output2_99560output2_99562*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_output2_layer_call_and_return_conditional_losses_995492!
output2/StatefulPartitionedCall¬
output1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0output1_99586output1_99588*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_output1_layer_call_and_return_conditional_losses_995752!
output1/StatefulPartitionedCallì
IdentityIdentity(output1/StatefulPartitionedCall:output:0#^block1a_bn/StatefulPartitionedCall'^block1a_dwconv/StatefulPartitionedCall+^block1a_project_bn/StatefulPartitionedCall-^block1a_project_conv/StatefulPartitionedCall*^block1a_se_expand/StatefulPartitionedCall*^block1a_se_reduce/StatefulPartitionedCall#^block1b_bn/StatefulPartitionedCall%^block1b_drop/StatefulPartitionedCall'^block1b_dwconv/StatefulPartitionedCall+^block1b_project_bn/StatefulPartitionedCall-^block1b_project_conv/StatefulPartitionedCall*^block1b_se_expand/StatefulPartitionedCall*^block1b_se_reduce/StatefulPartitionedCall^conv2d/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp ^output1/StatefulPartitionedCall ^output2/StatefulPartitionedCall ^output3/StatefulPartitionedCall ^output4/StatefulPartitionedCall ^stem_bn/StatefulPartitionedCall"^stem_conv/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityð

Identity_1Identity(output2/StatefulPartitionedCall:output:0#^block1a_bn/StatefulPartitionedCall'^block1a_dwconv/StatefulPartitionedCall+^block1a_project_bn/StatefulPartitionedCall-^block1a_project_conv/StatefulPartitionedCall*^block1a_se_expand/StatefulPartitionedCall*^block1a_se_reduce/StatefulPartitionedCall#^block1b_bn/StatefulPartitionedCall%^block1b_drop/StatefulPartitionedCall'^block1b_dwconv/StatefulPartitionedCall+^block1b_project_bn/StatefulPartitionedCall-^block1b_project_conv/StatefulPartitionedCall*^block1b_se_expand/StatefulPartitionedCall*^block1b_se_reduce/StatefulPartitionedCall^conv2d/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp ^output1/StatefulPartitionedCall ^output2/StatefulPartitionedCall ^output3/StatefulPartitionedCall ^output4/StatefulPartitionedCall ^stem_bn/StatefulPartitionedCall"^stem_conv/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1ð

Identity_2Identity(output3/StatefulPartitionedCall:output:0#^block1a_bn/StatefulPartitionedCall'^block1a_dwconv/StatefulPartitionedCall+^block1a_project_bn/StatefulPartitionedCall-^block1a_project_conv/StatefulPartitionedCall*^block1a_se_expand/StatefulPartitionedCall*^block1a_se_reduce/StatefulPartitionedCall#^block1b_bn/StatefulPartitionedCall%^block1b_drop/StatefulPartitionedCall'^block1b_dwconv/StatefulPartitionedCall+^block1b_project_bn/StatefulPartitionedCall-^block1b_project_conv/StatefulPartitionedCall*^block1b_se_expand/StatefulPartitionedCall*^block1b_se_reduce/StatefulPartitionedCall^conv2d/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp ^output1/StatefulPartitionedCall ^output2/StatefulPartitionedCall ^output3/StatefulPartitionedCall ^output4/StatefulPartitionedCall ^stem_bn/StatefulPartitionedCall"^stem_conv/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2ð

Identity_3Identity(output4/StatefulPartitionedCall:output:0#^block1a_bn/StatefulPartitionedCall'^block1a_dwconv/StatefulPartitionedCall+^block1a_project_bn/StatefulPartitionedCall-^block1a_project_conv/StatefulPartitionedCall*^block1a_se_expand/StatefulPartitionedCall*^block1a_se_reduce/StatefulPartitionedCall#^block1b_bn/StatefulPartitionedCall%^block1b_drop/StatefulPartitionedCall'^block1b_dwconv/StatefulPartitionedCall+^block1b_project_bn/StatefulPartitionedCall-^block1b_project_conv/StatefulPartitionedCall*^block1b_se_expand/StatefulPartitionedCall*^block1b_se_reduce/StatefulPartitionedCall^conv2d/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp ^output1/StatefulPartitionedCall ^output2/StatefulPartitionedCall ^output3/StatefulPartitionedCall ^output4/StatefulPartitionedCall ^stem_bn/StatefulPartitionedCall"^stem_conv/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*
_input_shapesô
ñ:ÿÿÿÿÿÿÿÿÿ¬¬:::::::::::::::::::::::::::::::::::::::::::::::::::::2H
"block1a_bn/StatefulPartitionedCall"block1a_bn/StatefulPartitionedCall2P
&block1a_dwconv/StatefulPartitionedCall&block1a_dwconv/StatefulPartitionedCall2X
*block1a_project_bn/StatefulPartitionedCall*block1a_project_bn/StatefulPartitionedCall2\
,block1a_project_conv/StatefulPartitionedCall,block1a_project_conv/StatefulPartitionedCall2V
)block1a_se_expand/StatefulPartitionedCall)block1a_se_expand/StatefulPartitionedCall2V
)block1a_se_reduce/StatefulPartitionedCall)block1a_se_reduce/StatefulPartitionedCall2H
"block1b_bn/StatefulPartitionedCall"block1b_bn/StatefulPartitionedCall2L
$block1b_drop/StatefulPartitionedCall$block1b_drop/StatefulPartitionedCall2P
&block1b_dwconv/StatefulPartitionedCall&block1b_dwconv/StatefulPartitionedCall2X
*block1b_project_bn/StatefulPartitionedCall*block1b_project_bn/StatefulPartitionedCall2\
,block1b_project_conv/StatefulPartitionedCall,block1b_project_conv/StatefulPartitionedCall2V
)block1b_se_expand/StatefulPartitionedCall)block1b_se_expand/StatefulPartitionedCall2V
)block1b_se_reduce/StatefulPartitionedCall)block1b_se_reduce/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2L
$normalization/Reshape/ReadVariableOp$normalization/Reshape/ReadVariableOp2P
&normalization/Reshape_1/ReadVariableOp&normalization/Reshape_1/ReadVariableOp2B
output1/StatefulPartitionedCalloutput1/StatefulPartitionedCall2B
output2/StatefulPartitionedCalloutput2/StatefulPartitionedCall2B
output3/StatefulPartitionedCalloutput3/StatefulPartitionedCall2B
output4/StatefulPartitionedCalloutput4/StatefulPartitionedCall2B
stem_bn/StatefulPartitionedCallstem_bn/StatefulPartitionedCall2F
!stem_conv/StatefulPartitionedCall!stem_conv/StatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬
!
_user_specified_name	input_1
µ
è
E__inference_block1b_bn_layer_call_and_return_conditional_losses_98417

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3ì
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
º
^
B__inference_flatten_layer_call_and_return_conditional_losses_99371

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
îá

@__inference_model_layer_call_and_return_conditional_losses_99926

inputs1
-normalization_reshape_readvariableop_resource3
/normalization_reshape_1_readvariableop_resource
stem_conv_99783
stem_bn_99786
stem_bn_99788
stem_bn_99790
stem_bn_99792
block1a_dwconv_99796
block1a_bn_99799
block1a_bn_99801
block1a_bn_99803
block1a_bn_99805
block1a_se_reduce_99811
block1a_se_reduce_99813
block1a_se_expand_99816
block1a_se_expand_99818
block1a_project_conv_99822
block1a_project_bn_99825
block1a_project_bn_99827
block1a_project_bn_99829
block1a_project_bn_99831
block1b_dwconv_99834
block1b_bn_99837
block1b_bn_99839
block1b_bn_99841
block1b_bn_99843
block1b_se_reduce_99849
block1b_se_reduce_99851
block1b_se_expand_99854
block1b_se_expand_99856
block1b_project_conv_99860
block1b_project_bn_99863
block1b_project_bn_99865
block1b_project_bn_99867
block1b_project_bn_99869
conv2d_99875
conv2d_99877
dense_3_99882
dense_3_99884
dense_2_99887
dense_2_99889
dense_1_99892
dense_1_99894
dense_99897
dense_99899
output4_99902
output4_99904
output3_99907
output3_99909
output2_99912
output2_99914
output1_99917
output1_99919
identity

identity_1

identity_2

identity_3¢"block1a_bn/StatefulPartitionedCall¢&block1a_dwconv/StatefulPartitionedCall¢*block1a_project_bn/StatefulPartitionedCall¢,block1a_project_conv/StatefulPartitionedCall¢)block1a_se_expand/StatefulPartitionedCall¢)block1a_se_reduce/StatefulPartitionedCall¢"block1b_bn/StatefulPartitionedCall¢$block1b_drop/StatefulPartitionedCall¢&block1b_dwconv/StatefulPartitionedCall¢*block1b_project_bn/StatefulPartitionedCall¢,block1b_project_conv/StatefulPartitionedCall¢)block1b_se_expand/StatefulPartitionedCall¢)block1b_se_reduce/StatefulPartitionedCall¢conv2d/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢$normalization/Reshape/ReadVariableOp¢&normalization/Reshape_1/ReadVariableOp¢output1/StatefulPartitionedCall¢output2/StatefulPartitionedCall¢output3/StatefulPartitionedCall¢output4/StatefulPartitionedCall¢stem_bn/StatefulPartitionedCall¢!stem_conv/StatefulPartitionedCalli
rescaling/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;2
rescaling/Cast/xm
rescaling/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rescaling/Cast_1/x
rescaling/mulMulinputsrescaling/Cast/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬2
rescaling/mul
rescaling/addAddV2rescaling/mul:z:0rescaling/Cast_1/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬2
rescaling/add¶
$normalization/Reshape/ReadVariableOpReadVariableOp-normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02&
$normalization/Reshape/ReadVariableOp
normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
normalization/Reshape/shape¾
normalization/ReshapeReshape,normalization/Reshape/ReadVariableOp:value:0$normalization/Reshape/shape:output:0*
T0*&
_output_shapes
:2
normalization/Reshape¼
&normalization/Reshape_1/ReadVariableOpReadVariableOp/normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization/Reshape_1/ReadVariableOp
normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
normalization/Reshape_1/shapeÆ
normalization/Reshape_1Reshape.normalization/Reshape_1/ReadVariableOp:value:0&normalization/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
normalization/Reshape_1
normalization/subSubrescaling/add:z:0normalization/Reshape:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬2
normalization/sub
normalization/SqrtSqrt normalization/Reshape_1:output:0*
T0*&
_output_shapes
:2
normalization/Sqrtw
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö32
normalization/Maximum/y¤
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*&
_output_shapes
:2
normalization/Maximum§
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬2
normalization/truedivÿ
stem_conv_pad/PartitionedCallPartitionedCallnormalization/truediv:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­­* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_stem_conv_pad_layer_call_and_return_conditional_losses_980042
stem_conv_pad/PartitionedCall­
!stem_conv/StatefulPartitionedCallStatefulPartitionedCall&stem_conv_pad/PartitionedCall:output:0stem_conv_99783*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_stem_conv_layer_call_and_return_conditional_losses_986252#
!stem_conv/StatefulPartitionedCallÜ
stem_bn/StatefulPartitionedCallStatefulPartitionedCall*stem_conv/StatefulPartitionedCall:output:0stem_bn_99786stem_bn_99788stem_bn_99790stem_bn_99792*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_stem_bn_layer_call_and_return_conditional_losses_986542!
stem_bn/StatefulPartitionedCall
stem_activation/PartitionedCallPartitionedCall(stem_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_stem_activation_layer_call_and_return_conditional_losses_987182!
stem_activation/PartitionedCallÃ
&block1a_dwconv/StatefulPartitionedCallStatefulPartitionedCall(stem_activation/PartitionedCall:output:0block1a_dwconv_99796*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_block1a_dwconv_layer_call_and_return_conditional_losses_981202(
&block1a_dwconv/StatefulPartitionedCallö
"block1a_bn/StatefulPartitionedCallStatefulPartitionedCall/block1a_dwconv/StatefulPartitionedCall:output:0block1a_bn_99799block1a_bn_99801block1a_bn_99803block1a_bn_99805*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_block1a_bn_layer_call_and_return_conditional_losses_987462$
"block1a_bn/StatefulPartitionedCall 
"block1a_activation/PartitionedCallPartitionedCall+block1a_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_block1a_activation_layer_call_and_return_conditional_losses_988102$
"block1a_activation/PartitionedCall
"block1a_se_squeeze/PartitionedCallPartitionedCall+block1a_activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_block1a_se_squeeze_layer_call_and_return_conditional_losses_982352$
"block1a_se_squeeze/PartitionedCall
"block1a_se_reshape/PartitionedCallPartitionedCall+block1a_se_squeeze/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_block1a_se_reshape_layer_call_and_return_conditional_losses_988332$
"block1a_se_reshape/PartitionedCallë
)block1a_se_reduce/StatefulPartitionedCallStatefulPartitionedCall+block1a_se_reshape/PartitionedCall:output:0block1a_se_reduce_99811block1a_se_reduce_99813*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_block1a_se_reduce_layer_call_and_return_conditional_losses_988572+
)block1a_se_reduce/StatefulPartitionedCallò
)block1a_se_expand/StatefulPartitionedCallStatefulPartitionedCall2block1a_se_reduce/StatefulPartitionedCall:output:0block1a_se_expand_99816block1a_se_expand_99818*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_block1a_se_expand_layer_call_and_return_conditional_losses_988842+
)block1a_se_expand/StatefulPartitionedCallÒ
!block1a_se_excite/PartitionedCallPartitionedCall+block1a_activation/PartitionedCall:output:02block1a_se_expand/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_block1a_se_excite_layer_call_and_return_conditional_losses_989062#
!block1a_se_excite/PartitionedCallÝ
,block1a_project_conv/StatefulPartitionedCallStatefulPartitionedCall*block1a_se_excite/PartitionedCall:output:0block1a_project_conv_99822*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_block1a_project_conv_layer_call_and_return_conditional_losses_989222.
,block1a_project_conv/StatefulPartitionedCall´
*block1a_project_bn/StatefulPartitionedCallStatefulPartitionedCall5block1a_project_conv/StatefulPartitionedCall:output:0block1a_project_bn_99825block1a_project_bn_99827block1a_project_bn_99829block1a_project_bn_99831*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_block1a_project_bn_layer_call_and_return_conditional_losses_989512,
*block1a_project_bn/StatefulPartitionedCallÎ
&block1b_dwconv/StatefulPartitionedCallStatefulPartitionedCall3block1a_project_bn/StatefulPartitionedCall:output:0block1b_dwconv_99834*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_block1b_dwconv_layer_call_and_return_conditional_losses_983512(
&block1b_dwconv/StatefulPartitionedCallö
"block1b_bn/StatefulPartitionedCallStatefulPartitionedCall/block1b_dwconv/StatefulPartitionedCall:output:0block1b_bn_99837block1b_bn_99839block1b_bn_99841block1b_bn_99843*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_block1b_bn_layer_call_and_return_conditional_losses_990252$
"block1b_bn/StatefulPartitionedCall 
"block1b_activation/PartitionedCallPartitionedCall+block1b_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_block1b_activation_layer_call_and_return_conditional_losses_990892$
"block1b_activation/PartitionedCall
"block1b_se_squeeze/PartitionedCallPartitionedCall+block1b_activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_block1b_se_squeeze_layer_call_and_return_conditional_losses_984662$
"block1b_se_squeeze/PartitionedCall
"block1b_se_reshape/PartitionedCallPartitionedCall+block1b_se_squeeze/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_block1b_se_reshape_layer_call_and_return_conditional_losses_991122$
"block1b_se_reshape/PartitionedCallë
)block1b_se_reduce/StatefulPartitionedCallStatefulPartitionedCall+block1b_se_reshape/PartitionedCall:output:0block1b_se_reduce_99849block1b_se_reduce_99851*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_block1b_se_reduce_layer_call_and_return_conditional_losses_991362+
)block1b_se_reduce/StatefulPartitionedCallò
)block1b_se_expand/StatefulPartitionedCallStatefulPartitionedCall2block1b_se_reduce/StatefulPartitionedCall:output:0block1b_se_expand_99854block1b_se_expand_99856*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_block1b_se_expand_layer_call_and_return_conditional_losses_991632+
)block1b_se_expand/StatefulPartitionedCallÒ
!block1b_se_excite/PartitionedCallPartitionedCall+block1b_activation/PartitionedCall:output:02block1b_se_expand/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_block1b_se_excite_layer_call_and_return_conditional_losses_991852#
!block1b_se_excite/PartitionedCallÝ
,block1b_project_conv/StatefulPartitionedCallStatefulPartitionedCall*block1b_se_excite/PartitionedCall:output:0block1b_project_conv_99860*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_block1b_project_conv_layer_call_and_return_conditional_losses_992012.
,block1b_project_conv/StatefulPartitionedCall´
*block1b_project_bn/StatefulPartitionedCallStatefulPartitionedCall5block1b_project_conv/StatefulPartitionedCall:output:0block1b_project_bn_99863block1b_project_bn_99865block1b_project_bn_99867block1b_project_bn_99869*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_block1b_project_bn_layer_call_and_return_conditional_losses_992302,
*block1b_project_bn/StatefulPartitionedCall®
$block1b_drop/StatefulPartitionedCallStatefulPartitionedCall3block1b_project_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_block1b_drop_layer_call_and_return_conditional_losses_993042&
$block1b_drop/StatefulPartitionedCallÃ
block1b_add/PartitionedCallPartitionedCall-block1b_drop/StatefulPartitionedCall:output:03block1a_project_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_block1b_add_layer_call_and_return_conditional_losses_993282
block1b_add/PartitionedCall
!average_pooling2d/PartitionedCallPartitionedCall$block1b_add/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_985782#
!average_pooling2d/PartitionedCall³
conv2d/StatefulPartitionedCallStatefulPartitionedCall*average_pooling2d/PartitionedCall:output:0conv2d_99875conv2d_99877*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_993482 
conv2d/StatefulPartitionedCall
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_985902
max_pooling2d/PartitionedCallñ
flatten/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_993712
flatten/PartitionedCall¦
dense_3/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_3_99882dense_3_99884*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_993902!
dense_3/StatefulPartitionedCall¦
dense_2/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_2_99887dense_2_99889*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_994172!
dense_2/StatefulPartitionedCall¦
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_99892dense_1_99894*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_994442!
dense_1/StatefulPartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_99897dense_99899*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_994712
dense/StatefulPartitionedCall®
output4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0output4_99902output4_99904*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_output4_layer_call_and_return_conditional_losses_994972!
output4/StatefulPartitionedCall®
output3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0output3_99907output3_99909*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_output3_layer_call_and_return_conditional_losses_995232!
output3/StatefulPartitionedCall®
output2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0output2_99912output2_99914*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_output2_layer_call_and_return_conditional_losses_995492!
output2/StatefulPartitionedCall¬
output1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0output1_99917output1_99919*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_output1_layer_call_and_return_conditional_losses_995752!
output1/StatefulPartitionedCallì
IdentityIdentity(output1/StatefulPartitionedCall:output:0#^block1a_bn/StatefulPartitionedCall'^block1a_dwconv/StatefulPartitionedCall+^block1a_project_bn/StatefulPartitionedCall-^block1a_project_conv/StatefulPartitionedCall*^block1a_se_expand/StatefulPartitionedCall*^block1a_se_reduce/StatefulPartitionedCall#^block1b_bn/StatefulPartitionedCall%^block1b_drop/StatefulPartitionedCall'^block1b_dwconv/StatefulPartitionedCall+^block1b_project_bn/StatefulPartitionedCall-^block1b_project_conv/StatefulPartitionedCall*^block1b_se_expand/StatefulPartitionedCall*^block1b_se_reduce/StatefulPartitionedCall^conv2d/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp ^output1/StatefulPartitionedCall ^output2/StatefulPartitionedCall ^output3/StatefulPartitionedCall ^output4/StatefulPartitionedCall ^stem_bn/StatefulPartitionedCall"^stem_conv/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityð

Identity_1Identity(output2/StatefulPartitionedCall:output:0#^block1a_bn/StatefulPartitionedCall'^block1a_dwconv/StatefulPartitionedCall+^block1a_project_bn/StatefulPartitionedCall-^block1a_project_conv/StatefulPartitionedCall*^block1a_se_expand/StatefulPartitionedCall*^block1a_se_reduce/StatefulPartitionedCall#^block1b_bn/StatefulPartitionedCall%^block1b_drop/StatefulPartitionedCall'^block1b_dwconv/StatefulPartitionedCall+^block1b_project_bn/StatefulPartitionedCall-^block1b_project_conv/StatefulPartitionedCall*^block1b_se_expand/StatefulPartitionedCall*^block1b_se_reduce/StatefulPartitionedCall^conv2d/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp ^output1/StatefulPartitionedCall ^output2/StatefulPartitionedCall ^output3/StatefulPartitionedCall ^output4/StatefulPartitionedCall ^stem_bn/StatefulPartitionedCall"^stem_conv/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1ð

Identity_2Identity(output3/StatefulPartitionedCall:output:0#^block1a_bn/StatefulPartitionedCall'^block1a_dwconv/StatefulPartitionedCall+^block1a_project_bn/StatefulPartitionedCall-^block1a_project_conv/StatefulPartitionedCall*^block1a_se_expand/StatefulPartitionedCall*^block1a_se_reduce/StatefulPartitionedCall#^block1b_bn/StatefulPartitionedCall%^block1b_drop/StatefulPartitionedCall'^block1b_dwconv/StatefulPartitionedCall+^block1b_project_bn/StatefulPartitionedCall-^block1b_project_conv/StatefulPartitionedCall*^block1b_se_expand/StatefulPartitionedCall*^block1b_se_reduce/StatefulPartitionedCall^conv2d/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp ^output1/StatefulPartitionedCall ^output2/StatefulPartitionedCall ^output3/StatefulPartitionedCall ^output4/StatefulPartitionedCall ^stem_bn/StatefulPartitionedCall"^stem_conv/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2ð

Identity_3Identity(output4/StatefulPartitionedCall:output:0#^block1a_bn/StatefulPartitionedCall'^block1a_dwconv/StatefulPartitionedCall+^block1a_project_bn/StatefulPartitionedCall-^block1a_project_conv/StatefulPartitionedCall*^block1a_se_expand/StatefulPartitionedCall*^block1a_se_reduce/StatefulPartitionedCall#^block1b_bn/StatefulPartitionedCall%^block1b_drop/StatefulPartitionedCall'^block1b_dwconv/StatefulPartitionedCall+^block1b_project_bn/StatefulPartitionedCall-^block1b_project_conv/StatefulPartitionedCall*^block1b_se_expand/StatefulPartitionedCall*^block1b_se_reduce/StatefulPartitionedCall^conv2d/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp ^output1/StatefulPartitionedCall ^output2/StatefulPartitionedCall ^output3/StatefulPartitionedCall ^output4/StatefulPartitionedCall ^stem_bn/StatefulPartitionedCall"^stem_conv/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*
_input_shapesô
ñ:ÿÿÿÿÿÿÿÿÿ¬¬:::::::::::::::::::::::::::::::::::::::::::::::::::::2H
"block1a_bn/StatefulPartitionedCall"block1a_bn/StatefulPartitionedCall2P
&block1a_dwconv/StatefulPartitionedCall&block1a_dwconv/StatefulPartitionedCall2X
*block1a_project_bn/StatefulPartitionedCall*block1a_project_bn/StatefulPartitionedCall2\
,block1a_project_conv/StatefulPartitionedCall,block1a_project_conv/StatefulPartitionedCall2V
)block1a_se_expand/StatefulPartitionedCall)block1a_se_expand/StatefulPartitionedCall2V
)block1a_se_reduce/StatefulPartitionedCall)block1a_se_reduce/StatefulPartitionedCall2H
"block1b_bn/StatefulPartitionedCall"block1b_bn/StatefulPartitionedCall2L
$block1b_drop/StatefulPartitionedCall$block1b_drop/StatefulPartitionedCall2P
&block1b_dwconv/StatefulPartitionedCall&block1b_dwconv/StatefulPartitionedCall2X
*block1b_project_bn/StatefulPartitionedCall*block1b_project_bn/StatefulPartitionedCall2\
,block1b_project_conv/StatefulPartitionedCall,block1b_project_conv/StatefulPartitionedCall2V
)block1b_se_expand/StatefulPartitionedCall)block1b_se_expand/StatefulPartitionedCall2V
)block1b_se_reduce/StatefulPartitionedCall)block1b_se_reduce/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2L
$normalization/Reshape/ReadVariableOp$normalization/Reshape/ReadVariableOp2P
&normalization/Reshape_1/ReadVariableOp&normalization/Reshape_1/ReadVariableOp2B
output1/StatefulPartitionedCalloutput1/StatefulPartitionedCall2B
output2/StatefulPartitionedCalloutput2/StatefulPartitionedCall2B
output3/StatefulPartitionedCalloutput3/StatefulPartitionedCall2B
output4/StatefulPartitionedCalloutput4/StatefulPartitionedCall2B
stem_bn/StatefulPartitionedCallstem_bn/StatefulPartitionedCall2F
!stem_conv/StatefulPartitionedCall!stem_conv/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬
 
_user_specified_nameinputs
Á
I
-__inference_block1b_drop_layer_call_fn_102114

inputs
identityÏ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_block1b_drop_layer_call_and_return_conditional_losses_993092
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È

+__inference_block1a_bn_layer_call_fn_101484

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_block1a_bn_layer_call_and_return_conditional_losses_987462
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ(::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
î
i
M__inference_block1b_se_reshape_layer_call_and_return_conditional_losses_99112

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3º
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
à
^
2__inference_block1b_se_excite_layer_call_fn_101941
inputs_0
inputs_1
identityá
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_block1b_se_excite_layer_call_and_return_conditional_losses_991852
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
Å	
ª
I__inference_block1b_dwconv_layer_call_and_return_conditional_losses_98351

inputs%
!depthwise_readvariableop_resource
identity¢depthwise/ReadVariableOp
depthwise/ReadVariableOpReadVariableOp!depthwise_readvariableop_resource*&
_output_shapes
:*
dtype02
depthwise/ReadVariableOp{
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
depthwise/Shape
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
depthwise/dilation_rateÍ
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
	depthwise
IdentityIdentitydepthwise:output:0^depthwise/ReadVariableOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:24
depthwise/ReadVariableOpdepthwise/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù
}
(__inference_output4_layer_call_fn_102312

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_output4_layer_call_and_return_conditional_losses_994972
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
à
p
F__inference_block1b_add_layer_call_and_return_conditional_losses_99328

inputs
inputs_1
identitya
addAddV2inputsinputs_1*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
adde
IdentityIdentityadd:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*M
_input_shapes<
::ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:YU
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø

å
L__inference_block1a_se_expand_layer_call_and_return_conditional_losses_98884

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
(*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2	
BiasAddi
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2	
Sigmoid
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
ï	
Û
B__inference_dense_3_layer_call_and_return_conditional_losses_99390

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø

å
L__inference_block1b_se_expand_layer_call_and_return_conditional_losses_99163

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddi
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoid
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë

D__inference_stem_conv_layer_call_and_return_conditional_losses_98625

inputs"
conv2d_readvariableop_resource
identity¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:(*
dtype02
Conv2D/ReadVariableOp¦
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
paddingVALID*
strides
2
Conv2D
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ­­:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­­
 
_user_specified_nameinputs
Þ
¾%
A__inference_model_layer_call_and_return_conditional_losses_100725

inputs1
-normalization_reshape_readvariableop_resource3
/normalization_reshape_1_readvariableop_resource,
(stem_conv_conv2d_readvariableop_resource#
stem_bn_readvariableop_resource%
!stem_bn_readvariableop_1_resource4
0stem_bn_fusedbatchnormv3_readvariableop_resource6
2stem_bn_fusedbatchnormv3_readvariableop_1_resource4
0block1a_dwconv_depthwise_readvariableop_resource&
"block1a_bn_readvariableop_resource(
$block1a_bn_readvariableop_1_resource7
3block1a_bn_fusedbatchnormv3_readvariableop_resource9
5block1a_bn_fusedbatchnormv3_readvariableop_1_resource4
0block1a_se_reduce_conv2d_readvariableop_resource5
1block1a_se_reduce_biasadd_readvariableop_resource4
0block1a_se_expand_conv2d_readvariableop_resource5
1block1a_se_expand_biasadd_readvariableop_resource7
3block1a_project_conv_conv2d_readvariableop_resource.
*block1a_project_bn_readvariableop_resource0
,block1a_project_bn_readvariableop_1_resource?
;block1a_project_bn_fusedbatchnormv3_readvariableop_resourceA
=block1a_project_bn_fusedbatchnormv3_readvariableop_1_resource4
0block1b_dwconv_depthwise_readvariableop_resource&
"block1b_bn_readvariableop_resource(
$block1b_bn_readvariableop_1_resource7
3block1b_bn_fusedbatchnormv3_readvariableop_resource9
5block1b_bn_fusedbatchnormv3_readvariableop_1_resource4
0block1b_se_reduce_conv2d_readvariableop_resource5
1block1b_se_reduce_biasadd_readvariableop_resource4
0block1b_se_expand_conv2d_readvariableop_resource5
1block1b_se_expand_biasadd_readvariableop_resource7
3block1b_project_conv_conv2d_readvariableop_resource.
*block1b_project_bn_readvariableop_resource0
,block1b_project_bn_readvariableop_1_resource?
;block1b_project_bn_fusedbatchnormv3_readvariableop_resourceA
=block1b_project_bn_fusedbatchnormv3_readvariableop_1_resource)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&output4_matmul_readvariableop_resource+
'output4_biasadd_readvariableop_resource*
&output3_matmul_readvariableop_resource+
'output3_biasadd_readvariableop_resource*
&output2_matmul_readvariableop_resource+
'output2_biasadd_readvariableop_resource*
&output1_matmul_readvariableop_resource+
'output1_biasadd_readvariableop_resource
identity

identity_1

identity_2

identity_3¢*block1a_bn/FusedBatchNormV3/ReadVariableOp¢,block1a_bn/FusedBatchNormV3/ReadVariableOp_1¢block1a_bn/ReadVariableOp¢block1a_bn/ReadVariableOp_1¢'block1a_dwconv/depthwise/ReadVariableOp¢2block1a_project_bn/FusedBatchNormV3/ReadVariableOp¢4block1a_project_bn/FusedBatchNormV3/ReadVariableOp_1¢!block1a_project_bn/ReadVariableOp¢#block1a_project_bn/ReadVariableOp_1¢*block1a_project_conv/Conv2D/ReadVariableOp¢(block1a_se_expand/BiasAdd/ReadVariableOp¢'block1a_se_expand/Conv2D/ReadVariableOp¢(block1a_se_reduce/BiasAdd/ReadVariableOp¢'block1a_se_reduce/Conv2D/ReadVariableOp¢*block1b_bn/FusedBatchNormV3/ReadVariableOp¢,block1b_bn/FusedBatchNormV3/ReadVariableOp_1¢block1b_bn/ReadVariableOp¢block1b_bn/ReadVariableOp_1¢'block1b_dwconv/depthwise/ReadVariableOp¢2block1b_project_bn/FusedBatchNormV3/ReadVariableOp¢4block1b_project_bn/FusedBatchNormV3/ReadVariableOp_1¢!block1b_project_bn/ReadVariableOp¢#block1b_project_bn/ReadVariableOp_1¢*block1b_project_conv/Conv2D/ReadVariableOp¢(block1b_se_expand/BiasAdd/ReadVariableOp¢'block1b_se_expand/Conv2D/ReadVariableOp¢(block1b_se_reduce/BiasAdd/ReadVariableOp¢'block1b_se_reduce/Conv2D/ReadVariableOp¢conv2d/BiasAdd/ReadVariableOp¢conv2d/Conv2D/ReadVariableOp¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOp¢dense_3/BiasAdd/ReadVariableOp¢dense_3/MatMul/ReadVariableOp¢$normalization/Reshape/ReadVariableOp¢&normalization/Reshape_1/ReadVariableOp¢output1/BiasAdd/ReadVariableOp¢output1/MatMul/ReadVariableOp¢output2/BiasAdd/ReadVariableOp¢output2/MatMul/ReadVariableOp¢output3/BiasAdd/ReadVariableOp¢output3/MatMul/ReadVariableOp¢output4/BiasAdd/ReadVariableOp¢output4/MatMul/ReadVariableOp¢'stem_bn/FusedBatchNormV3/ReadVariableOp¢)stem_bn/FusedBatchNormV3/ReadVariableOp_1¢stem_bn/ReadVariableOp¢stem_bn/ReadVariableOp_1¢stem_conv/Conv2D/ReadVariableOpi
rescaling/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;2
rescaling/Cast/xm
rescaling/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rescaling/Cast_1/x
rescaling/mulMulinputsrescaling/Cast/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬2
rescaling/mul
rescaling/addAddV2rescaling/mul:z:0rescaling/Cast_1/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬2
rescaling/add¶
$normalization/Reshape/ReadVariableOpReadVariableOp-normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02&
$normalization/Reshape/ReadVariableOp
normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
normalization/Reshape/shape¾
normalization/ReshapeReshape,normalization/Reshape/ReadVariableOp:value:0$normalization/Reshape/shape:output:0*
T0*&
_output_shapes
:2
normalization/Reshape¼
&normalization/Reshape_1/ReadVariableOpReadVariableOp/normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization/Reshape_1/ReadVariableOp
normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
normalization/Reshape_1/shapeÆ
normalization/Reshape_1Reshape.normalization/Reshape_1/ReadVariableOp:value:0&normalization/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
normalization/Reshape_1
normalization/subSubrescaling/add:z:0normalization/Reshape:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬2
normalization/sub
normalization/SqrtSqrt normalization/Reshape_1:output:0*
T0*&
_output_shapes
:2
normalization/Sqrtw
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö32
normalization/Maximum/y¤
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*&
_output_shapes
:2
normalization/Maximum§
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬2
normalization/truediv©
stem_conv_pad/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                               2
stem_conv_pad/Pad/paddings©
stem_conv_pad/PadPadnormalization/truediv:z:0#stem_conv_pad/Pad/paddings:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­­2
stem_conv_pad/Pad³
stem_conv/Conv2D/ReadVariableOpReadVariableOp(stem_conv_conv2d_readvariableop_resource*&
_output_shapes
:(*
dtype02!
stem_conv/Conv2D/ReadVariableOpØ
stem_conv/Conv2DConv2Dstem_conv_pad/Pad:output:0'stem_conv/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
paddingVALID*
strides
2
stem_conv/Conv2D
stem_bn/ReadVariableOpReadVariableOpstem_bn_readvariableop_resource*
_output_shapes
:(*
dtype02
stem_bn/ReadVariableOp
stem_bn/ReadVariableOp_1ReadVariableOp!stem_bn_readvariableop_1_resource*
_output_shapes
:(*
dtype02
stem_bn/ReadVariableOp_1¿
'stem_bn/FusedBatchNormV3/ReadVariableOpReadVariableOp0stem_bn_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:(*
dtype02)
'stem_bn/FusedBatchNormV3/ReadVariableOpÅ
)stem_bn/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp2stem_bn_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:(*
dtype02+
)stem_bn/FusedBatchNormV3/ReadVariableOp_1
stem_bn/FusedBatchNormV3FusedBatchNormV3stem_conv/Conv2D:output:0stem_bn/ReadVariableOp:value:0 stem_bn/ReadVariableOp_1:value:0/stem_bn/FusedBatchNormV3/ReadVariableOp:value:01stem_bn/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ(:(:(:(:(:*
epsilon%o:*
is_training( 2
stem_bn/FusedBatchNormV3
stem_activation/SigmoidSigmoidstem_bn/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
stem_activation/Sigmoid¨
stem_activation/mulMulstem_bn/FusedBatchNormV3:y:0stem_activation/Sigmoid:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
stem_activation/mul
stem_activation/IdentityIdentitystem_activation/mul:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
stem_activation/Identity
stem_activation/IdentityN	IdentityNstem_activation/mul:z:0stem_bn/FusedBatchNormV3:y:0*
T
2*,
_gradient_op_typeCustomGradient-100488*N
_output_shapes<
::ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(2
stem_activation/IdentityNË
'block1a_dwconv/depthwise/ReadVariableOpReadVariableOp0block1a_dwconv_depthwise_readvariableop_resource*&
_output_shapes
:(*
dtype02)
'block1a_dwconv/depthwise/ReadVariableOp
block1a_dwconv/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      (      2 
block1a_dwconv/depthwise/Shape¡
&block1a_dwconv/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2(
&block1a_dwconv/depthwise/dilation_rate
block1a_dwconv/depthwiseDepthwiseConv2dNative"stem_activation/IdentityN:output:0/block1a_dwconv/depthwise/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
paddingSAME*
strides
2
block1a_dwconv/depthwise
block1a_bn/ReadVariableOpReadVariableOp"block1a_bn_readvariableop_resource*
_output_shapes
:(*
dtype02
block1a_bn/ReadVariableOp
block1a_bn/ReadVariableOp_1ReadVariableOp$block1a_bn_readvariableop_1_resource*
_output_shapes
:(*
dtype02
block1a_bn/ReadVariableOp_1È
*block1a_bn/FusedBatchNormV3/ReadVariableOpReadVariableOp3block1a_bn_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:(*
dtype02,
*block1a_bn/FusedBatchNormV3/ReadVariableOpÎ
,block1a_bn/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5block1a_bn_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:(*
dtype02.
,block1a_bn/FusedBatchNormV3/ReadVariableOp_1©
block1a_bn/FusedBatchNormV3FusedBatchNormV3!block1a_dwconv/depthwise:output:0!block1a_bn/ReadVariableOp:value:0#block1a_bn/ReadVariableOp_1:value:02block1a_bn/FusedBatchNormV3/ReadVariableOp:value:04block1a_bn/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ(:(:(:(:(:*
epsilon%o:*
is_training( 2
block1a_bn/FusedBatchNormV3 
block1a_activation/SigmoidSigmoidblock1a_bn/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
block1a_activation/Sigmoid´
block1a_activation/mulMulblock1a_bn/FusedBatchNormV3:y:0block1a_activation/Sigmoid:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
block1a_activation/mul
block1a_activation/IdentityIdentityblock1a_activation/mul:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
block1a_activation/Identity
block1a_activation/IdentityN	IdentityNblock1a_activation/mul:z:0block1a_bn/FusedBatchNormV3:y:0*
T
2*,
_gradient_op_typeCustomGradient-100513*N
_output_shapes<
::ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(2
block1a_activation/IdentityN§
)block1a_se_squeeze/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2+
)block1a_se_squeeze/Mean/reduction_indicesÇ
block1a_se_squeeze/MeanMean%block1a_activation/IdentityN:output:02block1a_se_squeeze/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
block1a_se_squeeze/Mean
block1a_se_reshape/ShapeShape block1a_se_squeeze/Mean:output:0*
T0*
_output_shapes
:2
block1a_se_reshape/Shape
&block1a_se_reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&block1a_se_reshape/strided_slice/stack
(block1a_se_reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(block1a_se_reshape/strided_slice/stack_1
(block1a_se_reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(block1a_se_reshape/strided_slice/stack_2Ô
 block1a_se_reshape/strided_sliceStridedSlice!block1a_se_reshape/Shape:output:0/block1a_se_reshape/strided_slice/stack:output:01block1a_se_reshape/strided_slice/stack_1:output:01block1a_se_reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 block1a_se_reshape/strided_slice
"block1a_se_reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"block1a_se_reshape/Reshape/shape/1
"block1a_se_reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"block1a_se_reshape/Reshape/shape/2
"block1a_se_reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :(2$
"block1a_se_reshape/Reshape/shape/3¬
 block1a_se_reshape/Reshape/shapePack)block1a_se_reshape/strided_slice:output:0+block1a_se_reshape/Reshape/shape/1:output:0+block1a_se_reshape/Reshape/shape/2:output:0+block1a_se_reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2"
 block1a_se_reshape/Reshape/shapeÊ
block1a_se_reshape/ReshapeReshape block1a_se_squeeze/Mean:output:0)block1a_se_reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
block1a_se_reshape/ReshapeË
'block1a_se_reduce/Conv2D/ReadVariableOpReadVariableOp0block1a_se_reduce_conv2d_readvariableop_resource*&
_output_shapes
:(
*
dtype02)
'block1a_se_reduce/Conv2D/ReadVariableOpö
block1a_se_reduce/Conv2DConv2D#block1a_se_reshape/Reshape:output:0/block1a_se_reduce/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
paddingSAME*
strides
2
block1a_se_reduce/Conv2DÂ
(block1a_se_reduce/BiasAdd/ReadVariableOpReadVariableOp1block1a_se_reduce_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02*
(block1a_se_reduce/BiasAdd/ReadVariableOpÐ
block1a_se_reduce/BiasAddBiasAdd!block1a_se_reduce/Conv2D:output:00block1a_se_reduce/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
block1a_se_reduce/BiasAdd
block1a_se_reduce/SigmoidSigmoid"block1a_se_reduce/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
block1a_se_reduce/Sigmoid²
block1a_se_reduce/mulMul"block1a_se_reduce/BiasAdd:output:0block1a_se_reduce/Sigmoid:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
block1a_se_reduce/mul
block1a_se_reduce/IdentityIdentityblock1a_se_reduce/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
block1a_se_reduce/Identity
block1a_se_reduce/IdentityN	IdentityNblock1a_se_reduce/mul:z:0"block1a_se_reduce/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-100537*J
_output_shapes8
6:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
2
block1a_se_reduce/IdentityNË
'block1a_se_expand/Conv2D/ReadVariableOpReadVariableOp0block1a_se_expand_conv2d_readvariableop_resource*&
_output_shapes
:
(*
dtype02)
'block1a_se_expand/Conv2D/ReadVariableOp÷
block1a_se_expand/Conv2DConv2D$block1a_se_reduce/IdentityN:output:0/block1a_se_expand/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
paddingSAME*
strides
2
block1a_se_expand/Conv2DÂ
(block1a_se_expand/BiasAdd/ReadVariableOpReadVariableOp1block1a_se_expand_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02*
(block1a_se_expand/BiasAdd/ReadVariableOpÐ
block1a_se_expand/BiasAddBiasAdd!block1a_se_expand/Conv2D:output:00block1a_se_expand/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
block1a_se_expand/BiasAdd
block1a_se_expand/SigmoidSigmoid"block1a_se_expand/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
block1a_se_expand/Sigmoid·
block1a_se_excite/mulMul%block1a_activation/IdentityN:output:0block1a_se_expand/Sigmoid:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
block1a_se_excite/mulÔ
*block1a_project_conv/Conv2D/ReadVariableOpReadVariableOp3block1a_project_conv_conv2d_readvariableop_resource*&
_output_shapes
:(*
dtype02,
*block1a_project_conv/Conv2D/ReadVariableOp÷
block1a_project_conv/Conv2DConv2Dblock1a_se_excite/mul:z:02block1a_project_conv/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
block1a_project_conv/Conv2D­
!block1a_project_bn/ReadVariableOpReadVariableOp*block1a_project_bn_readvariableop_resource*
_output_shapes
:*
dtype02#
!block1a_project_bn/ReadVariableOp³
#block1a_project_bn/ReadVariableOp_1ReadVariableOp,block1a_project_bn_readvariableop_1_resource*
_output_shapes
:*
dtype02%
#block1a_project_bn/ReadVariableOp_1à
2block1a_project_bn/FusedBatchNormV3/ReadVariableOpReadVariableOp;block1a_project_bn_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype024
2block1a_project_bn/FusedBatchNormV3/ReadVariableOpæ
4block1a_project_bn/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp=block1a_project_bn_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype026
4block1a_project_bn/FusedBatchNormV3/ReadVariableOp_1Ü
#block1a_project_bn/FusedBatchNormV3FusedBatchNormV3$block1a_project_conv/Conv2D:output:0)block1a_project_bn/ReadVariableOp:value:0+block1a_project_bn/ReadVariableOp_1:value:0:block1a_project_bn/FusedBatchNormV3/ReadVariableOp:value:0<block1a_project_bn/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2%
#block1a_project_bn/FusedBatchNormV3Ë
'block1b_dwconv/depthwise/ReadVariableOpReadVariableOp0block1b_dwconv_depthwise_readvariableop_resource*&
_output_shapes
:*
dtype02)
'block1b_dwconv/depthwise/ReadVariableOp
block1b_dwconv/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2 
block1b_dwconv/depthwise/Shape¡
&block1b_dwconv/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2(
&block1b_dwconv/depthwise/dilation_rate
block1b_dwconv/depthwiseDepthwiseConv2dNative'block1a_project_bn/FusedBatchNormV3:y:0/block1b_dwconv/depthwise/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
block1b_dwconv/depthwise
block1b_bn/ReadVariableOpReadVariableOp"block1b_bn_readvariableop_resource*
_output_shapes
:*
dtype02
block1b_bn/ReadVariableOp
block1b_bn/ReadVariableOp_1ReadVariableOp$block1b_bn_readvariableop_1_resource*
_output_shapes
:*
dtype02
block1b_bn/ReadVariableOp_1È
*block1b_bn/FusedBatchNormV3/ReadVariableOpReadVariableOp3block1b_bn_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02,
*block1b_bn/FusedBatchNormV3/ReadVariableOpÎ
,block1b_bn/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5block1b_bn_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,block1b_bn/FusedBatchNormV3/ReadVariableOp_1©
block1b_bn/FusedBatchNormV3FusedBatchNormV3!block1b_dwconv/depthwise:output:0!block1b_bn/ReadVariableOp:value:0#block1b_bn/ReadVariableOp_1:value:02block1b_bn/FusedBatchNormV3/ReadVariableOp:value:04block1b_bn/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
block1b_bn/FusedBatchNormV3 
block1b_activation/SigmoidSigmoidblock1b_bn/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
block1b_activation/Sigmoid´
block1b_activation/mulMulblock1b_bn/FusedBatchNormV3:y:0block1b_activation/Sigmoid:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
block1b_activation/mul
block1b_activation/IdentityIdentityblock1b_activation/mul:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
block1b_activation/Identity
block1b_activation/IdentityN	IdentityNblock1b_activation/mul:z:0block1b_bn/FusedBatchNormV3:y:0*
T
2*,
_gradient_op_typeCustomGradient-100587*N
_output_shapes<
::ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ2
block1b_activation/IdentityN§
)block1b_se_squeeze/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2+
)block1b_se_squeeze/Mean/reduction_indicesÇ
block1b_se_squeeze/MeanMean%block1b_activation/IdentityN:output:02block1b_se_squeeze/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
block1b_se_squeeze/Mean
block1b_se_reshape/ShapeShape block1b_se_squeeze/Mean:output:0*
T0*
_output_shapes
:2
block1b_se_reshape/Shape
&block1b_se_reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&block1b_se_reshape/strided_slice/stack
(block1b_se_reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(block1b_se_reshape/strided_slice/stack_1
(block1b_se_reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(block1b_se_reshape/strided_slice/stack_2Ô
 block1b_se_reshape/strided_sliceStridedSlice!block1b_se_reshape/Shape:output:0/block1b_se_reshape/strided_slice/stack:output:01block1b_se_reshape/strided_slice/stack_1:output:01block1b_se_reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 block1b_se_reshape/strided_slice
"block1b_se_reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"block1b_se_reshape/Reshape/shape/1
"block1b_se_reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"block1b_se_reshape/Reshape/shape/2
"block1b_se_reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2$
"block1b_se_reshape/Reshape/shape/3¬
 block1b_se_reshape/Reshape/shapePack)block1b_se_reshape/strided_slice:output:0+block1b_se_reshape/Reshape/shape/1:output:0+block1b_se_reshape/Reshape/shape/2:output:0+block1b_se_reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2"
 block1b_se_reshape/Reshape/shapeÊ
block1b_se_reshape/ReshapeReshape block1b_se_squeeze/Mean:output:0)block1b_se_reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
block1b_se_reshape/ReshapeË
'block1b_se_reduce/Conv2D/ReadVariableOpReadVariableOp0block1b_se_reduce_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'block1b_se_reduce/Conv2D/ReadVariableOpö
block1b_se_reduce/Conv2DConv2D#block1b_se_reshape/Reshape:output:0/block1b_se_reduce/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
block1b_se_reduce/Conv2DÂ
(block1b_se_reduce/BiasAdd/ReadVariableOpReadVariableOp1block1b_se_reduce_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(block1b_se_reduce/BiasAdd/ReadVariableOpÐ
block1b_se_reduce/BiasAddBiasAdd!block1b_se_reduce/Conv2D:output:00block1b_se_reduce/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
block1b_se_reduce/BiasAdd
block1b_se_reduce/SigmoidSigmoid"block1b_se_reduce/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
block1b_se_reduce/Sigmoid²
block1b_se_reduce/mulMul"block1b_se_reduce/BiasAdd:output:0block1b_se_reduce/Sigmoid:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
block1b_se_reduce/mul
block1b_se_reduce/IdentityIdentityblock1b_se_reduce/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
block1b_se_reduce/Identity
block1b_se_reduce/IdentityN	IdentityNblock1b_se_reduce/mul:z:0"block1b_se_reduce/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-100611*J
_output_shapes8
6:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ2
block1b_se_reduce/IdentityNË
'block1b_se_expand/Conv2D/ReadVariableOpReadVariableOp0block1b_se_expand_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'block1b_se_expand/Conv2D/ReadVariableOp÷
block1b_se_expand/Conv2DConv2D$block1b_se_reduce/IdentityN:output:0/block1b_se_expand/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
block1b_se_expand/Conv2DÂ
(block1b_se_expand/BiasAdd/ReadVariableOpReadVariableOp1block1b_se_expand_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(block1b_se_expand/BiasAdd/ReadVariableOpÐ
block1b_se_expand/BiasAddBiasAdd!block1b_se_expand/Conv2D:output:00block1b_se_expand/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
block1b_se_expand/BiasAdd
block1b_se_expand/SigmoidSigmoid"block1b_se_expand/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
block1b_se_expand/Sigmoid·
block1b_se_excite/mulMul%block1b_activation/IdentityN:output:0block1b_se_expand/Sigmoid:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
block1b_se_excite/mulÔ
*block1b_project_conv/Conv2D/ReadVariableOpReadVariableOp3block1b_project_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02,
*block1b_project_conv/Conv2D/ReadVariableOp÷
block1b_project_conv/Conv2DConv2Dblock1b_se_excite/mul:z:02block1b_project_conv/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
block1b_project_conv/Conv2D­
!block1b_project_bn/ReadVariableOpReadVariableOp*block1b_project_bn_readvariableop_resource*
_output_shapes
:*
dtype02#
!block1b_project_bn/ReadVariableOp³
#block1b_project_bn/ReadVariableOp_1ReadVariableOp,block1b_project_bn_readvariableop_1_resource*
_output_shapes
:*
dtype02%
#block1b_project_bn/ReadVariableOp_1à
2block1b_project_bn/FusedBatchNormV3/ReadVariableOpReadVariableOp;block1b_project_bn_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype024
2block1b_project_bn/FusedBatchNormV3/ReadVariableOpæ
4block1b_project_bn/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp=block1b_project_bn_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype026
4block1b_project_bn/FusedBatchNormV3/ReadVariableOp_1Ü
#block1b_project_bn/FusedBatchNormV3FusedBatchNormV3$block1b_project_conv/Conv2D:output:0)block1b_project_bn/ReadVariableOp:value:0+block1b_project_bn/ReadVariableOp_1:value:0:block1b_project_bn/FusedBatchNormV3/ReadVariableOp:value:0<block1b_project_bn/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2%
#block1b_project_bn/FusedBatchNormV3
block1b_drop/ShapeShape'block1b_project_bn/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
block1b_drop/Shape
 block1b_drop/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 block1b_drop/strided_slice/stack
"block1b_drop/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"block1b_drop/strided_slice/stack_1
"block1b_drop/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"block1b_drop/strided_slice/stack_2°
block1b_drop/strided_sliceStridedSliceblock1b_drop/Shape:output:0)block1b_drop/strided_slice/stack:output:0+block1b_drop/strided_slice/stack_1:output:0+block1b_drop/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
block1b_drop/strided_slicep
block1b_drop/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
block1b_drop/packed/1p
block1b_drop/packed/2Const*
_output_shapes
: *
dtype0*
value	B :2
block1b_drop/packed/2p
block1b_drop/packed/3Const*
_output_shapes
: *
dtype0*
value	B :2
block1b_drop/packed/3å
block1b_drop/packedPack#block1b_drop/strided_slice:output:0block1b_drop/packed/1:output:0block1b_drop/packed/2:output:0block1b_drop/packed/3:output:0*
N*
T0*
_output_shapes
:2
block1b_drop/packed}
block1b_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *þ?2
block1b_drop/dropout/ConstÅ
block1b_drop/dropout/MulMul'block1b_project_bn/FusedBatchNormV3:y:0#block1b_drop/dropout/Const:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
block1b_drop/dropout/MulÜ
1block1b_drop/dropout/random_uniform/RandomUniformRandomUniformblock1b_drop/packed:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype023
1block1b_drop/dropout/random_uniform/RandomUniform
#block1b_drop/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Áü;2%
#block1b_drop/dropout/GreaterEqual/yú
!block1b_drop/dropout/GreaterEqualGreaterEqual:block1b_drop/dropout/random_uniform/RandomUniform:output:0,block1b_drop/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!block1b_drop/dropout/GreaterEqual®
block1b_drop/dropout/CastCast%block1b_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
block1b_drop/dropout/Cast¸
block1b_drop/dropout/Mul_1Mulblock1b_drop/dropout/Mul:z:0block1b_drop/dropout/Cast:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
block1b_drop/dropout/Mul_1°
block1b_add/addAddV2block1b_drop/dropout/Mul_1:z:0'block1a_project_bn/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
block1b_add/addÌ
average_pooling2d/AvgPoolAvgPoolblock1b_add/add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2
average_pooling2d/AvgPoolª
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOpÕ
conv2d/Conv2DConv2D"average_pooling2d/AvgPool:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d/Conv2D¡
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp¤
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d/BiasAdd¿
max_pooling2d/MaxPoolMaxPoolconv2d/BiasAdd:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
  2
flatten/Const
flatten/ReshapeReshapemax_pooling2d/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten/Reshape¦
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense_3/MatMul/ReadVariableOp
dense_3/MatMulMatMulflatten/Reshape:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_3/MatMul¤
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp¡
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_3/BiasAddp
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_3/Relu¦
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMulflatten/Reshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/MatMul¤
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp¡
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/Relu¦
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/MatMul¤
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp¡
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/Relu 
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

dense/Relu¥
output4/MatMul/ReadVariableOpReadVariableOp&output4_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
output4/MatMul/ReadVariableOp
output4/MatMulMatMuldense_3/Relu:activations:0%output4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
output4/MatMul¤
output4/BiasAdd/ReadVariableOpReadVariableOp'output4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
output4/BiasAdd/ReadVariableOp¡
output4/BiasAddBiasAddoutput4/MatMul:product:0&output4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
output4/BiasAdd¥
output3/MatMul/ReadVariableOpReadVariableOp&output3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
output3/MatMul/ReadVariableOp
output3/MatMulMatMuldense_2/Relu:activations:0%output3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
output3/MatMul¤
output3/BiasAdd/ReadVariableOpReadVariableOp'output3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
output3/BiasAdd/ReadVariableOp¡
output3/BiasAddBiasAddoutput3/MatMul:product:0&output3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
output3/BiasAdd¥
output2/MatMul/ReadVariableOpReadVariableOp&output2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
output2/MatMul/ReadVariableOp
output2/MatMulMatMuldense_1/Relu:activations:0%output2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
output2/MatMul¤
output2/BiasAdd/ReadVariableOpReadVariableOp'output2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
output2/BiasAdd/ReadVariableOp¡
output2/BiasAddBiasAddoutput2/MatMul:product:0&output2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
output2/BiasAdd¥
output1/MatMul/ReadVariableOpReadVariableOp&output1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
output1/MatMul/ReadVariableOp
output1/MatMulMatMuldense/Relu:activations:0%output1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
output1/MatMul¤
output1/BiasAdd/ReadVariableOpReadVariableOp'output1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
output1/BiasAdd/ReadVariableOp¡
output1/BiasAddBiasAddoutput1/MatMul:product:0&output1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
output1/BiasAddÅ
IdentityIdentityoutput1/BiasAdd:output:0+^block1a_bn/FusedBatchNormV3/ReadVariableOp-^block1a_bn/FusedBatchNormV3/ReadVariableOp_1^block1a_bn/ReadVariableOp^block1a_bn/ReadVariableOp_1(^block1a_dwconv/depthwise/ReadVariableOp3^block1a_project_bn/FusedBatchNormV3/ReadVariableOp5^block1a_project_bn/FusedBatchNormV3/ReadVariableOp_1"^block1a_project_bn/ReadVariableOp$^block1a_project_bn/ReadVariableOp_1+^block1a_project_conv/Conv2D/ReadVariableOp)^block1a_se_expand/BiasAdd/ReadVariableOp(^block1a_se_expand/Conv2D/ReadVariableOp)^block1a_se_reduce/BiasAdd/ReadVariableOp(^block1a_se_reduce/Conv2D/ReadVariableOp+^block1b_bn/FusedBatchNormV3/ReadVariableOp-^block1b_bn/FusedBatchNormV3/ReadVariableOp_1^block1b_bn/ReadVariableOp^block1b_bn/ReadVariableOp_1(^block1b_dwconv/depthwise/ReadVariableOp3^block1b_project_bn/FusedBatchNormV3/ReadVariableOp5^block1b_project_bn/FusedBatchNormV3/ReadVariableOp_1"^block1b_project_bn/ReadVariableOp$^block1b_project_bn/ReadVariableOp_1+^block1b_project_conv/Conv2D/ReadVariableOp)^block1b_se_expand/BiasAdd/ReadVariableOp(^block1b_se_expand/Conv2D/ReadVariableOp)^block1b_se_reduce/BiasAdd/ReadVariableOp(^block1b_se_reduce/Conv2D/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp^output1/BiasAdd/ReadVariableOp^output1/MatMul/ReadVariableOp^output2/BiasAdd/ReadVariableOp^output2/MatMul/ReadVariableOp^output3/BiasAdd/ReadVariableOp^output3/MatMul/ReadVariableOp^output4/BiasAdd/ReadVariableOp^output4/MatMul/ReadVariableOp(^stem_bn/FusedBatchNormV3/ReadVariableOp*^stem_bn/FusedBatchNormV3/ReadVariableOp_1^stem_bn/ReadVariableOp^stem_bn/ReadVariableOp_1 ^stem_conv/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÉ

Identity_1Identityoutput2/BiasAdd:output:0+^block1a_bn/FusedBatchNormV3/ReadVariableOp-^block1a_bn/FusedBatchNormV3/ReadVariableOp_1^block1a_bn/ReadVariableOp^block1a_bn/ReadVariableOp_1(^block1a_dwconv/depthwise/ReadVariableOp3^block1a_project_bn/FusedBatchNormV3/ReadVariableOp5^block1a_project_bn/FusedBatchNormV3/ReadVariableOp_1"^block1a_project_bn/ReadVariableOp$^block1a_project_bn/ReadVariableOp_1+^block1a_project_conv/Conv2D/ReadVariableOp)^block1a_se_expand/BiasAdd/ReadVariableOp(^block1a_se_expand/Conv2D/ReadVariableOp)^block1a_se_reduce/BiasAdd/ReadVariableOp(^block1a_se_reduce/Conv2D/ReadVariableOp+^block1b_bn/FusedBatchNormV3/ReadVariableOp-^block1b_bn/FusedBatchNormV3/ReadVariableOp_1^block1b_bn/ReadVariableOp^block1b_bn/ReadVariableOp_1(^block1b_dwconv/depthwise/ReadVariableOp3^block1b_project_bn/FusedBatchNormV3/ReadVariableOp5^block1b_project_bn/FusedBatchNormV3/ReadVariableOp_1"^block1b_project_bn/ReadVariableOp$^block1b_project_bn/ReadVariableOp_1+^block1b_project_conv/Conv2D/ReadVariableOp)^block1b_se_expand/BiasAdd/ReadVariableOp(^block1b_se_expand/Conv2D/ReadVariableOp)^block1b_se_reduce/BiasAdd/ReadVariableOp(^block1b_se_reduce/Conv2D/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp^output1/BiasAdd/ReadVariableOp^output1/MatMul/ReadVariableOp^output2/BiasAdd/ReadVariableOp^output2/MatMul/ReadVariableOp^output3/BiasAdd/ReadVariableOp^output3/MatMul/ReadVariableOp^output4/BiasAdd/ReadVariableOp^output4/MatMul/ReadVariableOp(^stem_bn/FusedBatchNormV3/ReadVariableOp*^stem_bn/FusedBatchNormV3/ReadVariableOp_1^stem_bn/ReadVariableOp^stem_bn/ReadVariableOp_1 ^stem_conv/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1É

Identity_2Identityoutput3/BiasAdd:output:0+^block1a_bn/FusedBatchNormV3/ReadVariableOp-^block1a_bn/FusedBatchNormV3/ReadVariableOp_1^block1a_bn/ReadVariableOp^block1a_bn/ReadVariableOp_1(^block1a_dwconv/depthwise/ReadVariableOp3^block1a_project_bn/FusedBatchNormV3/ReadVariableOp5^block1a_project_bn/FusedBatchNormV3/ReadVariableOp_1"^block1a_project_bn/ReadVariableOp$^block1a_project_bn/ReadVariableOp_1+^block1a_project_conv/Conv2D/ReadVariableOp)^block1a_se_expand/BiasAdd/ReadVariableOp(^block1a_se_expand/Conv2D/ReadVariableOp)^block1a_se_reduce/BiasAdd/ReadVariableOp(^block1a_se_reduce/Conv2D/ReadVariableOp+^block1b_bn/FusedBatchNormV3/ReadVariableOp-^block1b_bn/FusedBatchNormV3/ReadVariableOp_1^block1b_bn/ReadVariableOp^block1b_bn/ReadVariableOp_1(^block1b_dwconv/depthwise/ReadVariableOp3^block1b_project_bn/FusedBatchNormV3/ReadVariableOp5^block1b_project_bn/FusedBatchNormV3/ReadVariableOp_1"^block1b_project_bn/ReadVariableOp$^block1b_project_bn/ReadVariableOp_1+^block1b_project_conv/Conv2D/ReadVariableOp)^block1b_se_expand/BiasAdd/ReadVariableOp(^block1b_se_expand/Conv2D/ReadVariableOp)^block1b_se_reduce/BiasAdd/ReadVariableOp(^block1b_se_reduce/Conv2D/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp^output1/BiasAdd/ReadVariableOp^output1/MatMul/ReadVariableOp^output2/BiasAdd/ReadVariableOp^output2/MatMul/ReadVariableOp^output3/BiasAdd/ReadVariableOp^output3/MatMul/ReadVariableOp^output4/BiasAdd/ReadVariableOp^output4/MatMul/ReadVariableOp(^stem_bn/FusedBatchNormV3/ReadVariableOp*^stem_bn/FusedBatchNormV3/ReadVariableOp_1^stem_bn/ReadVariableOp^stem_bn/ReadVariableOp_1 ^stem_conv/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2É

Identity_3Identityoutput4/BiasAdd:output:0+^block1a_bn/FusedBatchNormV3/ReadVariableOp-^block1a_bn/FusedBatchNormV3/ReadVariableOp_1^block1a_bn/ReadVariableOp^block1a_bn/ReadVariableOp_1(^block1a_dwconv/depthwise/ReadVariableOp3^block1a_project_bn/FusedBatchNormV3/ReadVariableOp5^block1a_project_bn/FusedBatchNormV3/ReadVariableOp_1"^block1a_project_bn/ReadVariableOp$^block1a_project_bn/ReadVariableOp_1+^block1a_project_conv/Conv2D/ReadVariableOp)^block1a_se_expand/BiasAdd/ReadVariableOp(^block1a_se_expand/Conv2D/ReadVariableOp)^block1a_se_reduce/BiasAdd/ReadVariableOp(^block1a_se_reduce/Conv2D/ReadVariableOp+^block1b_bn/FusedBatchNormV3/ReadVariableOp-^block1b_bn/FusedBatchNormV3/ReadVariableOp_1^block1b_bn/ReadVariableOp^block1b_bn/ReadVariableOp_1(^block1b_dwconv/depthwise/ReadVariableOp3^block1b_project_bn/FusedBatchNormV3/ReadVariableOp5^block1b_project_bn/FusedBatchNormV3/ReadVariableOp_1"^block1b_project_bn/ReadVariableOp$^block1b_project_bn/ReadVariableOp_1+^block1b_project_conv/Conv2D/ReadVariableOp)^block1b_se_expand/BiasAdd/ReadVariableOp(^block1b_se_expand/Conv2D/ReadVariableOp)^block1b_se_reduce/BiasAdd/ReadVariableOp(^block1b_se_reduce/Conv2D/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp^output1/BiasAdd/ReadVariableOp^output1/MatMul/ReadVariableOp^output2/BiasAdd/ReadVariableOp^output2/MatMul/ReadVariableOp^output3/BiasAdd/ReadVariableOp^output3/MatMul/ReadVariableOp^output4/BiasAdd/ReadVariableOp^output4/MatMul/ReadVariableOp(^stem_bn/FusedBatchNormV3/ReadVariableOp*^stem_bn/FusedBatchNormV3/ReadVariableOp_1^stem_bn/ReadVariableOp^stem_bn/ReadVariableOp_1 ^stem_conv/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*
_input_shapesô
ñ:ÿÿÿÿÿÿÿÿÿ¬¬:::::::::::::::::::::::::::::::::::::::::::::::::::::2X
*block1a_bn/FusedBatchNormV3/ReadVariableOp*block1a_bn/FusedBatchNormV3/ReadVariableOp2\
,block1a_bn/FusedBatchNormV3/ReadVariableOp_1,block1a_bn/FusedBatchNormV3/ReadVariableOp_126
block1a_bn/ReadVariableOpblock1a_bn/ReadVariableOp2:
block1a_bn/ReadVariableOp_1block1a_bn/ReadVariableOp_12R
'block1a_dwconv/depthwise/ReadVariableOp'block1a_dwconv/depthwise/ReadVariableOp2h
2block1a_project_bn/FusedBatchNormV3/ReadVariableOp2block1a_project_bn/FusedBatchNormV3/ReadVariableOp2l
4block1a_project_bn/FusedBatchNormV3/ReadVariableOp_14block1a_project_bn/FusedBatchNormV3/ReadVariableOp_12F
!block1a_project_bn/ReadVariableOp!block1a_project_bn/ReadVariableOp2J
#block1a_project_bn/ReadVariableOp_1#block1a_project_bn/ReadVariableOp_12X
*block1a_project_conv/Conv2D/ReadVariableOp*block1a_project_conv/Conv2D/ReadVariableOp2T
(block1a_se_expand/BiasAdd/ReadVariableOp(block1a_se_expand/BiasAdd/ReadVariableOp2R
'block1a_se_expand/Conv2D/ReadVariableOp'block1a_se_expand/Conv2D/ReadVariableOp2T
(block1a_se_reduce/BiasAdd/ReadVariableOp(block1a_se_reduce/BiasAdd/ReadVariableOp2R
'block1a_se_reduce/Conv2D/ReadVariableOp'block1a_se_reduce/Conv2D/ReadVariableOp2X
*block1b_bn/FusedBatchNormV3/ReadVariableOp*block1b_bn/FusedBatchNormV3/ReadVariableOp2\
,block1b_bn/FusedBatchNormV3/ReadVariableOp_1,block1b_bn/FusedBatchNormV3/ReadVariableOp_126
block1b_bn/ReadVariableOpblock1b_bn/ReadVariableOp2:
block1b_bn/ReadVariableOp_1block1b_bn/ReadVariableOp_12R
'block1b_dwconv/depthwise/ReadVariableOp'block1b_dwconv/depthwise/ReadVariableOp2h
2block1b_project_bn/FusedBatchNormV3/ReadVariableOp2block1b_project_bn/FusedBatchNormV3/ReadVariableOp2l
4block1b_project_bn/FusedBatchNormV3/ReadVariableOp_14block1b_project_bn/FusedBatchNormV3/ReadVariableOp_12F
!block1b_project_bn/ReadVariableOp!block1b_project_bn/ReadVariableOp2J
#block1b_project_bn/ReadVariableOp_1#block1b_project_bn/ReadVariableOp_12X
*block1b_project_conv/Conv2D/ReadVariableOp*block1b_project_conv/Conv2D/ReadVariableOp2T
(block1b_se_expand/BiasAdd/ReadVariableOp(block1b_se_expand/BiasAdd/ReadVariableOp2R
'block1b_se_expand/Conv2D/ReadVariableOp'block1b_se_expand/Conv2D/ReadVariableOp2T
(block1b_se_reduce/BiasAdd/ReadVariableOp(block1b_se_reduce/BiasAdd/ReadVariableOp2R
'block1b_se_reduce/Conv2D/ReadVariableOp'block1b_se_reduce/Conv2D/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2L
$normalization/Reshape/ReadVariableOp$normalization/Reshape/ReadVariableOp2P
&normalization/Reshape_1/ReadVariableOp&normalization/Reshape_1/ReadVariableOp2@
output1/BiasAdd/ReadVariableOpoutput1/BiasAdd/ReadVariableOp2>
output1/MatMul/ReadVariableOpoutput1/MatMul/ReadVariableOp2@
output2/BiasAdd/ReadVariableOpoutput2/BiasAdd/ReadVariableOp2>
output2/MatMul/ReadVariableOpoutput2/MatMul/ReadVariableOp2@
output3/BiasAdd/ReadVariableOpoutput3/BiasAdd/ReadVariableOp2>
output3/MatMul/ReadVariableOpoutput3/MatMul/ReadVariableOp2@
output4/BiasAdd/ReadVariableOpoutput4/BiasAdd/ReadVariableOp2>
output4/MatMul/ReadVariableOpoutput4/MatMul/ReadVariableOp2R
'stem_bn/FusedBatchNormV3/ReadVariableOp'stem_bn/FusedBatchNormV3/ReadVariableOp2V
)stem_bn/FusedBatchNormV3/ReadVariableOp_1)stem_bn/FusedBatchNormV3/ReadVariableOp_120
stem_bn/ReadVariableOpstem_bn/ReadVariableOp24
stem_bn/ReadVariableOp_1stem_bn/ReadVariableOp_12B
stem_conv/Conv2D/ReadVariableOpstem_conv/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬
 
_user_specified_nameinputs
Õ
ª
O__inference_block1b_project_conv_layer_call_and_return_conditional_losses_99201

inputs"
conv2d_readvariableop_resource
identity¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Ü
C__inference_output2_layer_call_and_return_conditional_losses_102265

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
ñ
N__inference_block1b_project_bn_layer_call_and_return_conditional_losses_101973

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3ì
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Û
}
(__inference_dense_2_layer_call_fn_102216

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_994172
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü
N
2__inference_block1a_se_squeeze_layer_call_fn_98241

inputs
identityÔ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_block1a_se_squeeze_layer_call_and_return_conditional_losses_982352
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
½
ð
M__inference_block1a_project_bn_layer_call_and_return_conditional_losses_98330

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3ì
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
à
^
2__inference_block1a_se_excite_layer_call_fn_101588
inputs_0
inputs_1
identityá
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_block1a_se_excite_layer_call_and_return_conditional_losses_989062
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
"
_user_specified_name
inputs/1

i
K__inference_stem_activation_layer_call_and_return_conditional_losses_101368

inputs

identity_1a
SigmoidSigmoidinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2	
Sigmoidb
mulMulinputsSigmoid:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
mule
IdentityIdentitymul:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity¿
	IdentityN	IdentityNmul:z:0inputs*
T
2*,
_gradient_op_typeCustomGradient-101361*N
_output_shapes<
::ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(2
	IdentityNt

Identity_1IdentityIdentityN:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity_1"!

identity_1Identity_1:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs

è
M__inference_block1b_se_reduce_layer_call_and_return_conditional_losses_101900

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource

identity_1¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddi
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidj
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulc
IdentityIdentitymul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÅ
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-101893*J
_output_shapes8
6:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ2
	IdentityN£

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×
{
&__inference_dense_layer_call_fn_102176

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallð
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_994712
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷
|
'__inference_conv2d_layer_call_fn_102145

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_993482
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


(__inference_stem_bn_layer_call_fn_101283

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¦
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_stem_bn_layer_call_and_return_conditional_losses_980682
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
µ
è
E__inference_block1a_bn_layer_call_and_return_conditional_losses_98186

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:(*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:(*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:(*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:(*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(:(:(:(:(:*
epsilon%o:*
is_training( 2
FusedBatchNormV3ì
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ë
serving_default×
E
input_1:
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ¬¬;
output10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ;
output20
StatefulPartitionedCall:1ÿÿÿÿÿÿÿÿÿ;
output30
StatefulPartitionedCall:2ÿÿÿÿÿÿÿÿÿ;
output40
StatefulPartitionedCall:3ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:Ï	
¢Ï
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer-10
layer-11
layer_with_weights-5
layer-12
layer_with_weights-6
layer-13
layer-14
layer_with_weights-7
layer-15
layer_with_weights-8
layer-16
layer_with_weights-9
layer-17
layer_with_weights-10
layer-18
layer-19
layer-20
layer-21
layer_with_weights-11
layer-22
layer_with_weights-12
layer-23
layer-24
layer_with_weights-13
layer-25
layer_with_weights-14
layer-26
layer-27
layer-28
layer-29
layer_with_weights-15
layer-30
 layer-31
!layer-32
"layer_with_weights-16
"layer-33
#layer_with_weights-17
#layer-34
$layer_with_weights-18
$layer-35
%layer_with_weights-19
%layer-36
&layer_with_weights-20
&layer-37
'layer_with_weights-21
'layer-38
(layer_with_weights-22
(layer-39
)layer_with_weights-23
)layer-40
*	optimizer
+loss
,regularization_losses
-	variables
.trainable_variables
/	keras_api
0
signatures
__call__
_default_save_signature
+&call_and_return_all_conditional_losses"Ä
_tf_keras_networkúÃ{"class_name": "Functional", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 300, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Rescaling", "config": {"name": "rescaling", "trainable": false, "dtype": "float32", "scale": 0.00392156862745098, "offset": 0.0}, "name": "rescaling", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "normalization", "trainable": false, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [3]}}, "name": "normalization", "inbound_nodes": [[["rescaling", 0, 0, {}]]]}, {"class_name": "ZeroPadding2D", "config": {"name": "stem_conv_pad", "trainable": false, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [0, 1]}, {"class_name": "__tuple__", "items": [0, 1]}]}, "data_format": "channels_last"}, "name": "stem_conv_pad", "inbound_nodes": [[["normalization", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "stem_conv", "trainable": false, "dtype": "float32", "filters": 40, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_out", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "stem_conv", "inbound_nodes": [[["stem_conv_pad", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "stem_bn", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "stem_bn", "inbound_nodes": [[["stem_conv", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "stem_activation", "trainable": false, "dtype": "float32", "activation": "swish"}, "name": "stem_activation", "inbound_nodes": [[["stem_bn", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "block1a_dwconv", "trainable": false, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_out", "distribution": "truncated_normal", "seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "block1a_dwconv", "inbound_nodes": [[["stem_activation", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "block1a_bn", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "block1a_bn", "inbound_nodes": [[["block1a_dwconv", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "block1a_activation", "trainable": false, "dtype": "float32", "activation": "swish"}, "name": "block1a_activation", "inbound_nodes": [[["block1a_bn", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "block1a_se_squeeze", "trainable": false, "dtype": "float32", "data_format": "channels_last"}, "name": "block1a_se_squeeze", "inbound_nodes": [[["block1a_activation", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "block1a_se_reshape", "trainable": false, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 1, 40]}}, "name": "block1a_se_reshape", "inbound_nodes": [[["block1a_se_squeeze", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1a_se_reduce", "trainable": false, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_out", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1a_se_reduce", "inbound_nodes": [[["block1a_se_reshape", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1a_se_expand", "trainable": false, "dtype": "float32", "filters": 40, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_out", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1a_se_expand", "inbound_nodes": [[["block1a_se_reduce", 0, 0, {}]]]}, {"class_name": "Multiply", "config": {"name": "block1a_se_excite", "trainable": false, "dtype": "float32"}, "name": "block1a_se_excite", "inbound_nodes": [[["block1a_activation", 0, 0, {}], ["block1a_se_expand", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1a_project_conv", "trainable": false, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_out", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1a_project_conv", "inbound_nodes": [[["block1a_se_excite", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "block1a_project_bn", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "block1a_project_bn", "inbound_nodes": [[["block1a_project_conv", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "block1b_dwconv", "trainable": false, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_out", "distribution": "truncated_normal", "seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "block1b_dwconv", "inbound_nodes": [[["block1a_project_bn", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "block1b_bn", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "block1b_bn", "inbound_nodes": [[["block1b_dwconv", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "block1b_activation", "trainable": false, "dtype": "float32", "activation": "swish"}, "name": "block1b_activation", "inbound_nodes": [[["block1b_bn", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "block1b_se_squeeze", "trainable": false, "dtype": "float32", "data_format": "channels_last"}, "name": "block1b_se_squeeze", "inbound_nodes": [[["block1b_activation", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "block1b_se_reshape", "trainable": false, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 1, 24]}}, "name": "block1b_se_reshape", "inbound_nodes": [[["block1b_se_squeeze", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1b_se_reduce", "trainable": false, "dtype": "float32", "filters": 6, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_out", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1b_se_reduce", "inbound_nodes": [[["block1b_se_reshape", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1b_se_expand", "trainable": false, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_out", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1b_se_expand", "inbound_nodes": [[["block1b_se_reduce", 0, 0, {}]]]}, {"class_name": "Multiply", "config": {"name": "block1b_se_excite", "trainable": false, "dtype": "float32"}, "name": "block1b_se_excite", "inbound_nodes": [[["block1b_activation", 0, 0, {}], ["block1b_se_expand", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1b_project_conv", "trainable": false, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_out", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1b_project_conv", "inbound_nodes": [[["block1b_se_excite", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "block1b_project_bn", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "block1b_project_bn", "inbound_nodes": [[["block1b_project_conv", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "block1b_drop", "trainable": false, "dtype": "float32", "rate": 0.007692307692307693, "noise_shape": {"class_name": "__tuple__", "items": [null, 1, 1, 1]}, "seed": null}, "name": "block1b_drop", "inbound_nodes": [[["block1b_project_bn", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "block1b_add", "trainable": false, "dtype": "float32"}, "name": "block1b_add", "inbound_nodes": [[["block1b_drop", 0, 0, {}], ["block1a_project_bn", 0, 0, {}]]]}, {"class_name": "AveragePooling2D", "config": {"name": "average_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [5, 5]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [5, 5]}, "data_format": "channels_last"}, "name": "average_pooling2d", "inbound_nodes": [[["block1b_add", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["average_pooling2d", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["max_pooling2d", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 5, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 5, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 5, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 5, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output1", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output2", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output3", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output3", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output4", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output4", "inbound_nodes": [[["dense_3", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["output1", 0, 0], ["output2", 0, 0], ["output3", 0, 0], ["output4", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 300, 300, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 300, 300, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 300, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Rescaling", "config": {"name": "rescaling", "trainable": false, "dtype": "float32", "scale": 0.00392156862745098, "offset": 0.0}, "name": "rescaling", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "normalization", "trainable": false, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [3]}}, "name": "normalization", "inbound_nodes": [[["rescaling", 0, 0, {}]]]}, {"class_name": "ZeroPadding2D", "config": {"name": "stem_conv_pad", "trainable": false, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [0, 1]}, {"class_name": "__tuple__", "items": [0, 1]}]}, "data_format": "channels_last"}, "name": "stem_conv_pad", "inbound_nodes": [[["normalization", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "stem_conv", "trainable": false, "dtype": "float32", "filters": 40, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_out", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "stem_conv", "inbound_nodes": [[["stem_conv_pad", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "stem_bn", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "stem_bn", "inbound_nodes": [[["stem_conv", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "stem_activation", "trainable": false, "dtype": "float32", "activation": "swish"}, "name": "stem_activation", "inbound_nodes": [[["stem_bn", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "block1a_dwconv", "trainable": false, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_out", "distribution": "truncated_normal", "seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "block1a_dwconv", "inbound_nodes": [[["stem_activation", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "block1a_bn", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "block1a_bn", "inbound_nodes": [[["block1a_dwconv", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "block1a_activation", "trainable": false, "dtype": "float32", "activation": "swish"}, "name": "block1a_activation", "inbound_nodes": [[["block1a_bn", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "block1a_se_squeeze", "trainable": false, "dtype": "float32", "data_format": "channels_last"}, "name": "block1a_se_squeeze", "inbound_nodes": [[["block1a_activation", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "block1a_se_reshape", "trainable": false, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 1, 40]}}, "name": "block1a_se_reshape", "inbound_nodes": [[["block1a_se_squeeze", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1a_se_reduce", "trainable": false, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_out", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1a_se_reduce", "inbound_nodes": [[["block1a_se_reshape", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1a_se_expand", "trainable": false, "dtype": "float32", "filters": 40, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_out", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1a_se_expand", "inbound_nodes": [[["block1a_se_reduce", 0, 0, {}]]]}, {"class_name": "Multiply", "config": {"name": "block1a_se_excite", "trainable": false, "dtype": "float32"}, "name": "block1a_se_excite", "inbound_nodes": [[["block1a_activation", 0, 0, {}], ["block1a_se_expand", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1a_project_conv", "trainable": false, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_out", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1a_project_conv", "inbound_nodes": [[["block1a_se_excite", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "block1a_project_bn", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "block1a_project_bn", "inbound_nodes": [[["block1a_project_conv", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "block1b_dwconv", "trainable": false, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_out", "distribution": "truncated_normal", "seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "block1b_dwconv", "inbound_nodes": [[["block1a_project_bn", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "block1b_bn", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "block1b_bn", "inbound_nodes": [[["block1b_dwconv", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "block1b_activation", "trainable": false, "dtype": "float32", "activation": "swish"}, "name": "block1b_activation", "inbound_nodes": [[["block1b_bn", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "block1b_se_squeeze", "trainable": false, "dtype": "float32", "data_format": "channels_last"}, "name": "block1b_se_squeeze", "inbound_nodes": [[["block1b_activation", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "block1b_se_reshape", "trainable": false, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 1, 24]}}, "name": "block1b_se_reshape", "inbound_nodes": [[["block1b_se_squeeze", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1b_se_reduce", "trainable": false, "dtype": "float32", "filters": 6, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_out", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1b_se_reduce", "inbound_nodes": [[["block1b_se_reshape", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1b_se_expand", "trainable": false, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_out", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1b_se_expand", "inbound_nodes": [[["block1b_se_reduce", 0, 0, {}]]]}, {"class_name": "Multiply", "config": {"name": "block1b_se_excite", "trainable": false, "dtype": "float32"}, "name": "block1b_se_excite", "inbound_nodes": [[["block1b_activation", 0, 0, {}], ["block1b_se_expand", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1b_project_conv", "trainable": false, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_out", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1b_project_conv", "inbound_nodes": [[["block1b_se_excite", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "block1b_project_bn", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "block1b_project_bn", "inbound_nodes": [[["block1b_project_conv", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "block1b_drop", "trainable": false, "dtype": "float32", "rate": 0.007692307692307693, "noise_shape": {"class_name": "__tuple__", "items": [null, 1, 1, 1]}, "seed": null}, "name": "block1b_drop", "inbound_nodes": [[["block1b_project_bn", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "block1b_add", "trainable": false, "dtype": "float32"}, "name": "block1b_add", "inbound_nodes": [[["block1b_drop", 0, 0, {}], ["block1a_project_bn", 0, 0, {}]]]}, {"class_name": "AveragePooling2D", "config": {"name": "average_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [5, 5]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [5, 5]}, "data_format": "channels_last"}, "name": "average_pooling2d", "inbound_nodes": [[["block1b_add", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["average_pooling2d", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["max_pooling2d", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 5, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 5, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 5, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 5, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output1", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output2", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output3", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output3", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output4", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output4", "inbound_nodes": [[["dense_3", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["output1", 0, 0], ["output2", 0, 0], ["output3", 0, 0], ["output4", 0, 0]]}}, "training_config": {"loss": {"output1": {"class_name": "MeanSquaredError", "config": {"reduction": "auto", "name": "mean_squared_error"}}, "output2": {"class_name": "MeanSquaredError", "config": {"reduction": "auto", "name": "mean_squared_error"}}, "output3": {"class_name": "MeanSquaredError", "config": {"reduction": "auto", "name": "mean_squared_error"}}, "output4": {"class_name": "MeanSquaredError", "config": {"reduction": "auto", "name": "mean_squared_error"}}}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ý"ú
_tf_keras_input_layerÚ{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 300, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 300, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
é
1	keras_api"×
_tf_keras_layer½{"class_name": "Rescaling", "name": "rescaling", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "rescaling", "trainable": false, "dtype": "float32", "scale": 0.00392156862745098, "offset": 0.0}}

2state_variables
3_broadcast_shape
4mean
5variance
	6count
7	keras_api"»
_tf_keras_layer¡{"class_name": "Normalization", "name": "normalization", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "normalization", "trainable": false, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [3]}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 300, 300, 3]}}

8regularization_losses
9	variables
:trainable_variables
;	keras_api
__call__
+&call_and_return_all_conditional_losses"ö
_tf_keras_layerÜ{"class_name": "ZeroPadding2D", "name": "stem_conv_pad", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "stem_conv_pad", "trainable": false, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [0, 1]}, {"class_name": "__tuple__", "items": [0, 1]}]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
¹


<kernel
=regularization_losses
>	variables
?trainable_variables
@	keras_api
__call__
+&call_and_return_all_conditional_losses"	
_tf_keras_layer	{"class_name": "Conv2D", "name": "stem_conv", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "stem_conv", "trainable": false, "dtype": "float32", "filters": 40, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_out", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 301, 301, 3]}}
¤	
Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
__call__
+&call_and_return_all_conditional_losses"Î
_tf_keras_layer´{"class_name": "BatchNormalization", "name": "stem_bn", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "stem_bn", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 40}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 150, 150, 40]}}
à
Jregularization_losses
K	variables
Ltrainable_variables
M	keras_api
__call__
+&call_and_return_all_conditional_losses"Ï
_tf_keras_layerµ{"class_name": "Activation", "name": "stem_activation", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "stem_activation", "trainable": false, "dtype": "float32", "activation": "swish"}}
è

Ndepthwise_kernel
Oregularization_losses
P	variables
Qtrainable_variables
R	keras_api
__call__
+&call_and_return_all_conditional_losses"Á	
_tf_keras_layer§	{"class_name": "DepthwiseConv2D", "name": "block1a_dwconv", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block1a_dwconv", "trainable": false, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_out", "distribution": "truncated_normal", "seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 40}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 150, 150, 40]}}
ª	
Saxis
	Tgamma
Ubeta
Vmoving_mean
Wmoving_variance
Xregularization_losses
Y	variables
Ztrainable_variables
[	keras_api
__call__
+&call_and_return_all_conditional_losses"Ô
_tf_keras_layerº{"class_name": "BatchNormalization", "name": "block1a_bn", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block1a_bn", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 40}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 150, 150, 40]}}
æ
\regularization_losses
]	variables
^trainable_variables
_	keras_api
__call__
+&call_and_return_all_conditional_losses"Õ
_tf_keras_layer»{"class_name": "Activation", "name": "block1a_activation", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block1a_activation", "trainable": false, "dtype": "float32", "activation": "swish"}}

`regularization_losses
a	variables
btrainable_variables
c	keras_api
__call__
+&call_and_return_all_conditional_losses"ú
_tf_keras_layerà{"class_name": "GlobalAveragePooling2D", "name": "block1a_se_squeeze", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block1a_se_squeeze", "trainable": false, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}

dregularization_losses
e	variables
ftrainable_variables
g	keras_api
 __call__
+¡&call_and_return_all_conditional_losses"ý
_tf_keras_layerã{"class_name": "Reshape", "name": "block1a_se_reshape", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block1a_se_reshape", "trainable": false, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 1, 40]}}}
Î


hkernel
ibias
jregularization_losses
k	variables
ltrainable_variables
m	keras_api
¢__call__
+£&call_and_return_all_conditional_losses"§	
_tf_keras_layer	{"class_name": "Conv2D", "name": "block1a_se_reduce", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block1a_se_reduce", "trainable": false, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_out", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 40}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 1, 40]}}
Ð


nkernel
obias
pregularization_losses
q	variables
rtrainable_variables
s	keras_api
¤__call__
+¥&call_and_return_all_conditional_losses"©	
_tf_keras_layer	{"class_name": "Conv2D", "name": "block1a_se_expand", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block1a_se_expand", "trainable": false, "dtype": "float32", "filters": 40, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_out", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 1, 10]}}
Ú
tregularization_losses
u	variables
vtrainable_variables
w	keras_api
¦__call__
+§&call_and_return_all_conditional_losses"É
_tf_keras_layer¯{"class_name": "Multiply", "name": "block1a_se_excite", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block1a_se_excite", "trainable": false, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 150, 150, 40]}, {"class_name": "TensorShape", "items": [null, 1, 1, 40]}]}
Ð


xkernel
yregularization_losses
z	variables
{trainable_variables
|	keras_api
¨__call__
+©&call_and_return_all_conditional_losses"³	
_tf_keras_layer	{"class_name": "Conv2D", "name": "block1a_project_conv", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block1a_project_conv", "trainable": false, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_out", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 40}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 150, 150, 40]}}
À	
}axis
	~gamma
beta
moving_mean
moving_variance
regularization_losses
	variables
trainable_variables
	keras_api
ª__call__
+«&call_and_return_all_conditional_losses"ä
_tf_keras_layerÊ{"class_name": "BatchNormalization", "name": "block1a_project_bn", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block1a_project_bn", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 24}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 150, 150, 24]}}
í

depthwise_kernel
regularization_losses
	variables
trainable_variables
	keras_api
¬__call__
+­&call_and_return_all_conditional_losses"Á	
_tf_keras_layer§	{"class_name": "DepthwiseConv2D", "name": "block1b_dwconv", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block1b_dwconv", "trainable": false, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_out", "distribution": "truncated_normal", "seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 24}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 150, 150, 24]}}
³	
	axis

gamma
	beta
moving_mean
moving_variance
regularization_losses
	variables
trainable_variables
	keras_api
®__call__
+¯&call_and_return_all_conditional_losses"Ô
_tf_keras_layerº{"class_name": "BatchNormalization", "name": "block1b_bn", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block1b_bn", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 24}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 150, 150, 24]}}
ê
regularization_losses
	variables
trainable_variables
	keras_api
°__call__
+±&call_and_return_all_conditional_losses"Õ
_tf_keras_layer»{"class_name": "Activation", "name": "block1b_activation", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block1b_activation", "trainable": false, "dtype": "float32", "activation": "swish"}}

regularization_losses
	variables
trainable_variables
	keras_api
²__call__
+³&call_and_return_all_conditional_losses"ú
_tf_keras_layerà{"class_name": "GlobalAveragePooling2D", "name": "block1b_se_squeeze", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block1b_se_squeeze", "trainable": false, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}

regularization_losses
	variables
trainable_variables
	keras_api
´__call__
+µ&call_and_return_all_conditional_losses"ý
_tf_keras_layerã{"class_name": "Reshape", "name": "block1b_se_reshape", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block1b_se_reshape", "trainable": false, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 1, 24]}}}
Ó

 kernel
	¡bias
¢regularization_losses
£	variables
¤trainable_variables
¥	keras_api
¶__call__
+·&call_and_return_all_conditional_losses"¦	
_tf_keras_layer	{"class_name": "Conv2D", "name": "block1b_se_reduce", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block1b_se_reduce", "trainable": false, "dtype": "float32", "filters": 6, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_out", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 24}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 1, 24]}}
Ô

¦kernel
	§bias
¨regularization_losses
©	variables
ªtrainable_variables
«	keras_api
¸__call__
+¹&call_and_return_all_conditional_losses"§	
_tf_keras_layer	{"class_name": "Conv2D", "name": "block1b_se_expand", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block1b_se_expand", "trainable": false, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_out", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 1, 6]}}
Þ
¬regularization_losses
­	variables
®trainable_variables
¯	keras_api
º__call__
+»&call_and_return_all_conditional_losses"É
_tf_keras_layer¯{"class_name": "Multiply", "name": "block1b_se_excite", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block1b_se_excite", "trainable": false, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 150, 150, 24]}, {"class_name": "TensorShape", "items": [null, 1, 1, 24]}]}
Õ

°kernel
±regularization_losses
²	variables
³trainable_variables
´	keras_api
¼__call__
+½&call_and_return_all_conditional_losses"³	
_tf_keras_layer	{"class_name": "Conv2D", "name": "block1b_project_conv", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block1b_project_conv", "trainable": false, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_out", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 24}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 150, 150, 24]}}
Ã	
	µaxis

¶gamma
	·beta
¸moving_mean
¹moving_variance
ºregularization_losses
»	variables
¼trainable_variables
½	keras_api
¾__call__
+¿&call_and_return_all_conditional_losses"ä
_tf_keras_layerÊ{"class_name": "BatchNormalization", "name": "block1b_project_bn", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block1b_project_bn", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 24}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 150, 150, 24]}}
µ
¾regularization_losses
¿	variables
Àtrainable_variables
Á	keras_api
À__call__
+Á&call_and_return_all_conditional_losses" 
_tf_keras_layer{"class_name": "Dropout", "name": "block1b_drop", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block1b_drop", "trainable": false, "dtype": "float32", "rate": 0.007692307692307693, "noise_shape": {"class_name": "__tuple__", "items": [null, 1, 1, 1]}, "seed": null}}
Ñ
Âregularization_losses
Ã	variables
Ätrainable_variables
Å	keras_api
Â__call__
+Ã&call_and_return_all_conditional_losses"¼
_tf_keras_layer¢{"class_name": "Add", "name": "block1b_add", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block1b_add", "trainable": false, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 150, 150, 24]}, {"class_name": "TensorShape", "items": [null, 150, 150, 24]}]}

Æregularization_losses
Ç	variables
Ètrainable_variables
É	keras_api
Ä__call__
+Å&call_and_return_all_conditional_losses"ø
_tf_keras_layerÞ{"class_name": "AveragePooling2D", "name": "average_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [5, 5]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [5, 5]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ù	
Êkernel
	Ëbias
Ìregularization_losses
Í	variables
Îtrainable_variables
Ï	keras_api
Æ__call__
+Ç&call_and_return_all_conditional_losses"Ì
_tf_keras_layer²{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 24}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 30, 24]}}

Ðregularization_losses
Ñ	variables
Òtrainable_variables
Ó	keras_api
È__call__
+É&call_and_return_all_conditional_losses"ì
_tf_keras_layerÒ{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
è
Ôregularization_losses
Õ	variables
Ötrainable_variables
×	keras_api
Ê__call__
+Ë&call_and_return_all_conditional_losses"Ó
_tf_keras_layer¹{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
÷
Økernel
	Ùbias
Úregularization_losses
Û	variables
Ütrainable_variables
Ý	keras_api
Ì__call__
+Í&call_and_return_all_conditional_losses"Ê
_tf_keras_layer°{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 5, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2704}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2704]}}
û
Þkernel
	ßbias
àregularization_losses
á	variables
âtrainable_variables
ã	keras_api
Î__call__
+Ï&call_and_return_all_conditional_losses"Î
_tf_keras_layer´{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 5, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2704}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2704]}}
û
äkernel
	åbias
æregularization_losses
ç	variables
ètrainable_variables
é	keras_api
Ð__call__
+Ñ&call_and_return_all_conditional_losses"Î
_tf_keras_layer´{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 5, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2704}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2704]}}
û
êkernel
	ëbias
ìregularization_losses
í	variables
îtrainable_variables
ï	keras_api
Ò__call__
+Ó&call_and_return_all_conditional_losses"Î
_tf_keras_layer´{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 5, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2704}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2704]}}
÷
ðkernel
	ñbias
òregularization_losses
ó	variables
ôtrainable_variables
õ	keras_api
Ô__call__
+Õ&call_and_return_all_conditional_losses"Ê
_tf_keras_layer°{"class_name": "Dense", "name": "output1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "output1", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 5}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5]}}
÷
ökernel
	÷bias
øregularization_losses
ù	variables
útrainable_variables
û	keras_api
Ö__call__
+×&call_and_return_all_conditional_losses"Ê
_tf_keras_layer°{"class_name": "Dense", "name": "output2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "output2", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 5}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5]}}
÷
ükernel
	ýbias
þregularization_losses
ÿ	variables
trainable_variables
	keras_api
Ø__call__
+Ù&call_and_return_all_conditional_losses"Ê
_tf_keras_layer°{"class_name": "Dense", "name": "output3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "output3", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 5}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5]}}
÷
kernel
	bias
regularization_losses
	variables
trainable_variables
	keras_api
Ú__call__
+Û&call_and_return_all_conditional_losses"Ê
_tf_keras_layer°{"class_name": "Dense", "name": "output4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "output4", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 5}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5]}}
ä
	iter
beta_1
beta_2

decay
learning_rate	Êmé	Ëmê	Ømë	Ùmì	Þmí	ßmî	ämï	åmð	êmñ	ëmò	ðmó	ñmô	ömõ	÷mö	üm÷	ýmø	mù	mú	Êvû	Ëvü	Øvý	Ùvþ	Þvÿ	ßv	äv	åv	êv	ëv	ðv	ñv	öv	÷v	üv	ýv	v	v"
	optimizer
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
è
40
51
62
<3
B4
C5
D6
E7
N8
T9
U10
V11
W12
h13
i14
n15
o16
x17
~18
19
20
21
22
23
24
25
26
 27
¡28
¦29
§30
°31
¶32
·33
¸34
¹35
Ê36
Ë37
Ø38
Ù39
Þ40
ß41
ä42
å43
ê44
ë45
ð46
ñ47
ö48
÷49
ü50
ý51
52
53"
trackable_list_wrapper
¸
Ê0
Ë1
Ø2
Ù3
Þ4
ß5
ä6
å7
ê8
ë9
ð10
ñ11
ö12
÷13
ü14
ý15
16
17"
trackable_list_wrapper
Ó
,regularization_losses
-	variables
.trainable_variables
non_trainable_variables
 layer_regularization_losses
layer_metrics
layers
metrics
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
Üserving_default"
signature_map
"
_generic_user_object
C
4mean
5variance
	6count"
trackable_dict_wrapper
 "
trackable_list_wrapper
:2normalization/mean
": 2normalization/variance
:	 2normalization/count
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
8regularization_losses
9	variables
non_trainable_variables
 layer_regularization_losses
:trainable_variables
layer_metrics
layers
metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
*:((2stem_conv/kernel
 "
trackable_list_wrapper
'
<0"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
=regularization_losses
>	variables
non_trainable_variables
 layer_regularization_losses
?trainable_variables
layer_metrics
layers
metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:(2stem_bn/gamma
:(2stem_bn/beta
#:!( (2stem_bn/moving_mean
':%( (2stem_bn/moving_variance
 "
trackable_list_wrapper
<
B0
C1
D2
E3"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Fregularization_losses
G	variables
non_trainable_variables
 layer_regularization_losses
Htrainable_variables
layer_metrics
layers
 metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Jregularization_losses
K	variables
¡non_trainable_variables
 ¢layer_regularization_losses
Ltrainable_variables
£layer_metrics
¤layers
¥metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
9:7(2block1a_dwconv/depthwise_kernel
 "
trackable_list_wrapper
'
N0"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Oregularization_losses
P	variables
¦non_trainable_variables
 §layer_regularization_losses
Qtrainable_variables
¨layer_metrics
©layers
ªmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:(2block1a_bn/gamma
:(2block1a_bn/beta
&:$( (2block1a_bn/moving_mean
*:(( (2block1a_bn/moving_variance
 "
trackable_list_wrapper
<
T0
U1
V2
W3"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Xregularization_losses
Y	variables
«non_trainable_variables
 ¬layer_regularization_losses
Ztrainable_variables
­layer_metrics
®layers
¯metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
\regularization_losses
]	variables
°non_trainable_variables
 ±layer_regularization_losses
^trainable_variables
²layer_metrics
³layers
´metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
`regularization_losses
a	variables
µnon_trainable_variables
 ¶layer_regularization_losses
btrainable_variables
·layer_metrics
¸layers
¹metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
dregularization_losses
e	variables
ºnon_trainable_variables
 »layer_regularization_losses
ftrainable_variables
¼layer_metrics
½layers
¾metrics
 __call__
+¡&call_and_return_all_conditional_losses
'¡"call_and_return_conditional_losses"
_generic_user_object
2:0(
2block1a_se_reduce/kernel
$:"
2block1a_se_reduce/bias
 "
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
jregularization_losses
k	variables
¿non_trainable_variables
 Àlayer_regularization_losses
ltrainable_variables
Álayer_metrics
Âlayers
Ãmetrics
¢__call__
+£&call_and_return_all_conditional_losses
'£"call_and_return_conditional_losses"
_generic_user_object
2:0
(2block1a_se_expand/kernel
$:"(2block1a_se_expand/bias
 "
trackable_list_wrapper
.
n0
o1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
pregularization_losses
q	variables
Änon_trainable_variables
 Ålayer_regularization_losses
rtrainable_variables
Ælayer_metrics
Çlayers
Èmetrics
¤__call__
+¥&call_and_return_all_conditional_losses
'¥"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
tregularization_losses
u	variables
Énon_trainable_variables
 Êlayer_regularization_losses
vtrainable_variables
Ëlayer_metrics
Ìlayers
Ímetrics
¦__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
5:3(2block1a_project_conv/kernel
 "
trackable_list_wrapper
'
x0"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
yregularization_losses
z	variables
Înon_trainable_variables
 Ïlayer_regularization_losses
{trainable_variables
Ðlayer_metrics
Ñlayers
Òmetrics
¨__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
&:$2block1a_project_bn/gamma
%:#2block1a_project_bn/beta
.:, (2block1a_project_bn/moving_mean
2:0 (2"block1a_project_bn/moving_variance
 "
trackable_list_wrapper
>
~0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
regularization_losses
	variables
Ónon_trainable_variables
 Ôlayer_regularization_losses
trainable_variables
Õlayer_metrics
Ölayers
×metrics
ª__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
9:72block1b_dwconv/depthwise_kernel
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
regularization_losses
	variables
Ønon_trainable_variables
 Ùlayer_regularization_losses
trainable_variables
Úlayer_metrics
Ûlayers
Ümetrics
¬__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:2block1b_bn/gamma
:2block1b_bn/beta
&:$ (2block1b_bn/moving_mean
*:( (2block1b_bn/moving_variance
 "
trackable_list_wrapper
@
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
regularization_losses
	variables
Ýnon_trainable_variables
 Þlayer_regularization_losses
trainable_variables
ßlayer_metrics
àlayers
ámetrics
®__call__
+¯&call_and_return_all_conditional_losses
'¯"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
regularization_losses
	variables
ânon_trainable_variables
 ãlayer_regularization_losses
trainable_variables
älayer_metrics
ålayers
æmetrics
°__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
regularization_losses
	variables
çnon_trainable_variables
 èlayer_regularization_losses
trainable_variables
élayer_metrics
êlayers
ëmetrics
²__call__
+³&call_and_return_all_conditional_losses
'³"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
regularization_losses
	variables
ìnon_trainable_variables
 ílayer_regularization_losses
trainable_variables
îlayer_metrics
ïlayers
ðmetrics
´__call__
+µ&call_and_return_all_conditional_losses
'µ"call_and_return_conditional_losses"
_generic_user_object
2:02block1b_se_reduce/kernel
$:"2block1b_se_reduce/bias
 "
trackable_list_wrapper
0
 0
¡1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¢regularization_losses
£	variables
ñnon_trainable_variables
 òlayer_regularization_losses
¤trainable_variables
ólayer_metrics
ôlayers
õmetrics
¶__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses"
_generic_user_object
2:02block1b_se_expand/kernel
$:"2block1b_se_expand/bias
 "
trackable_list_wrapper
0
¦0
§1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¨regularization_losses
©	variables
önon_trainable_variables
 ÷layer_regularization_losses
ªtrainable_variables
ølayer_metrics
ùlayers
úmetrics
¸__call__
+¹&call_and_return_all_conditional_losses
'¹"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¬regularization_losses
­	variables
ûnon_trainable_variables
 ülayer_regularization_losses
®trainable_variables
ýlayer_metrics
þlayers
ÿmetrics
º__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses"
_generic_user_object
5:32block1b_project_conv/kernel
 "
trackable_list_wrapper
(
°0"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
±regularization_losses
²	variables
non_trainable_variables
 layer_regularization_losses
³trainable_variables
layer_metrics
layers
metrics
¼__call__
+½&call_and_return_all_conditional_losses
'½"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
&:$2block1b_project_bn/gamma
%:#2block1b_project_bn/beta
.:, (2block1b_project_bn/moving_mean
2:0 (2"block1b_project_bn/moving_variance
 "
trackable_list_wrapper
@
¶0
·1
¸2
¹3"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ºregularization_losses
»	variables
non_trainable_variables
 layer_regularization_losses
¼trainable_variables
layer_metrics
layers
metrics
¾__call__
+¿&call_and_return_all_conditional_losses
'¿"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¾regularization_losses
¿	variables
non_trainable_variables
 layer_regularization_losses
Àtrainable_variables
layer_metrics
layers
metrics
À__call__
+Á&call_and_return_all_conditional_losses
'Á"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Âregularization_losses
Ã	variables
non_trainable_variables
 layer_regularization_losses
Ätrainable_variables
layer_metrics
layers
metrics
Â__call__
+Ã&call_and_return_all_conditional_losses
'Ã"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Æregularization_losses
Ç	variables
non_trainable_variables
 layer_regularization_losses
Ètrainable_variables
layer_metrics
layers
metrics
Ä__call__
+Å&call_and_return_all_conditional_losses
'Å"call_and_return_conditional_losses"
_generic_user_object
':%2conv2d/kernel
:2conv2d/bias
 "
trackable_list_wrapper
0
Ê0
Ë1"
trackable_list_wrapper
0
Ê0
Ë1"
trackable_list_wrapper
¸
Ìregularization_losses
Í	variables
non_trainable_variables
 layer_regularization_losses
Îtrainable_variables
layer_metrics
layers
metrics
Æ__call__
+Ç&call_and_return_all_conditional_losses
'Ç"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ðregularization_losses
Ñ	variables
non_trainable_variables
 layer_regularization_losses
Òtrainable_variables
 layer_metrics
¡layers
¢metrics
È__call__
+É&call_and_return_all_conditional_losses
'É"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ôregularization_losses
Õ	variables
£non_trainable_variables
 ¤layer_regularization_losses
Ötrainable_variables
¥layer_metrics
¦layers
§metrics
Ê__call__
+Ë&call_and_return_all_conditional_losses
'Ë"call_and_return_conditional_losses"
_generic_user_object
:	2dense/kernel
:2
dense/bias
 "
trackable_list_wrapper
0
Ø0
Ù1"
trackable_list_wrapper
0
Ø0
Ù1"
trackable_list_wrapper
¸
Úregularization_losses
Û	variables
¨non_trainable_variables
 ©layer_regularization_losses
Ütrainable_variables
ªlayer_metrics
«layers
¬metrics
Ì__call__
+Í&call_and_return_all_conditional_losses
'Í"call_and_return_conditional_losses"
_generic_user_object
!:	2dense_1/kernel
:2dense_1/bias
 "
trackable_list_wrapper
0
Þ0
ß1"
trackable_list_wrapper
0
Þ0
ß1"
trackable_list_wrapper
¸
àregularization_losses
á	variables
­non_trainable_variables
 ®layer_regularization_losses
âtrainable_variables
¯layer_metrics
°layers
±metrics
Î__call__
+Ï&call_and_return_all_conditional_losses
'Ï"call_and_return_conditional_losses"
_generic_user_object
!:	2dense_2/kernel
:2dense_2/bias
 "
trackable_list_wrapper
0
ä0
å1"
trackable_list_wrapper
0
ä0
å1"
trackable_list_wrapper
¸
æregularization_losses
ç	variables
²non_trainable_variables
 ³layer_regularization_losses
ètrainable_variables
´layer_metrics
µlayers
¶metrics
Ð__call__
+Ñ&call_and_return_all_conditional_losses
'Ñ"call_and_return_conditional_losses"
_generic_user_object
!:	2dense_3/kernel
:2dense_3/bias
 "
trackable_list_wrapper
0
ê0
ë1"
trackable_list_wrapper
0
ê0
ë1"
trackable_list_wrapper
¸
ìregularization_losses
í	variables
·non_trainable_variables
 ¸layer_regularization_losses
îtrainable_variables
¹layer_metrics
ºlayers
»metrics
Ò__call__
+Ó&call_and_return_all_conditional_losses
'Ó"call_and_return_conditional_losses"
_generic_user_object
 :2output1/kernel
:2output1/bias
 "
trackable_list_wrapper
0
ð0
ñ1"
trackable_list_wrapper
0
ð0
ñ1"
trackable_list_wrapper
¸
òregularization_losses
ó	variables
¼non_trainable_variables
 ½layer_regularization_losses
ôtrainable_variables
¾layer_metrics
¿layers
Àmetrics
Ô__call__
+Õ&call_and_return_all_conditional_losses
'Õ"call_and_return_conditional_losses"
_generic_user_object
 :2output2/kernel
:2output2/bias
 "
trackable_list_wrapper
0
ö0
÷1"
trackable_list_wrapper
0
ö0
÷1"
trackable_list_wrapper
¸
øregularization_losses
ù	variables
Ánon_trainable_variables
 Âlayer_regularization_losses
útrainable_variables
Ãlayer_metrics
Älayers
Åmetrics
Ö__call__
+×&call_and_return_all_conditional_losses
'×"call_and_return_conditional_losses"
_generic_user_object
 :2output3/kernel
:2output3/bias
 "
trackable_list_wrapper
0
ü0
ý1"
trackable_list_wrapper
0
ü0
ý1"
trackable_list_wrapper
¸
þregularization_losses
ÿ	variables
Ænon_trainable_variables
 Çlayer_regularization_losses
trainable_variables
Èlayer_metrics
Élayers
Êmetrics
Ø__call__
+Ù&call_and_return_all_conditional_losses
'Ù"call_and_return_conditional_losses"
_generic_user_object
 :2output4/kernel
:2output4/bias
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
regularization_losses
	variables
Ënon_trainable_variables
 Ìlayer_regularization_losses
trainable_variables
Ílayer_metrics
Îlayers
Ïmetrics
Ú__call__
+Û&call_and_return_all_conditional_losses
'Û"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
Æ
40
51
62
<3
B4
C5
D6
E7
N8
T9
U10
V11
W12
h13
i14
n15
o16
x17
~18
19
20
21
22
23
24
25
26
 27
¡28
¦29
§30
°31
¶32
·33
¸34
¹35"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Þ
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40"
trackable_list_wrapper
H
Ð0
Ñ1
Ò2
Ó3
Ô4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
<0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
B0
C1
D2
E3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
N0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
T0
U1
V2
W3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
n0
o1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
x0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
>
~0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
@
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
 0
¡1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
¦0
§1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
°0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
@
¶0
·1
¸2
¹3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¿

Õtotal

Öcount
×	variables
Ø	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
Ï

Ùtotal

Úcount
Û	variables
Ü	keras_api"
_tf_keras_metricz{"class_name": "Mean", "name": "output1_loss", "dtype": "float32", "config": {"name": "output1_loss", "dtype": "float32"}}
Ï

Ýtotal

Þcount
ß	variables
à	keras_api"
_tf_keras_metricz{"class_name": "Mean", "name": "output2_loss", "dtype": "float32", "config": {"name": "output2_loss", "dtype": "float32"}}
Ï

átotal

âcount
ã	variables
ä	keras_api"
_tf_keras_metricz{"class_name": "Mean", "name": "output3_loss", "dtype": "float32", "config": {"name": "output3_loss", "dtype": "float32"}}
Ï

åtotal

æcount
ç	variables
è	keras_api"
_tf_keras_metricz{"class_name": "Mean", "name": "output4_loss", "dtype": "float32", "config": {"name": "output4_loss", "dtype": "float32"}}
:  (2total
:  (2count
0
Õ0
Ö1"
trackable_list_wrapper
.
×	variables"
_generic_user_object
:  (2total
:  (2count
0
Ù0
Ú1"
trackable_list_wrapper
.
Û	variables"
_generic_user_object
:  (2total
:  (2count
0
Ý0
Þ1"
trackable_list_wrapper
.
ß	variables"
_generic_user_object
:  (2total
:  (2count
0
á0
â1"
trackable_list_wrapper
.
ã	variables"
_generic_user_object
:  (2total
:  (2count
0
å0
æ1"
trackable_list_wrapper
.
ç	variables"
_generic_user_object
,:*2Adam/conv2d/kernel/m
:2Adam/conv2d/bias/m
$:"	2Adam/dense/kernel/m
:2Adam/dense/bias/m
&:$	2Adam/dense_1/kernel/m
:2Adam/dense_1/bias/m
&:$	2Adam/dense_2/kernel/m
:2Adam/dense_2/bias/m
&:$	2Adam/dense_3/kernel/m
:2Adam/dense_3/bias/m
%:#2Adam/output1/kernel/m
:2Adam/output1/bias/m
%:#2Adam/output2/kernel/m
:2Adam/output2/bias/m
%:#2Adam/output3/kernel/m
:2Adam/output3/bias/m
%:#2Adam/output4/kernel/m
:2Adam/output4/bias/m
,:*2Adam/conv2d/kernel/v
:2Adam/conv2d/bias/v
$:"	2Adam/dense/kernel/v
:2Adam/dense/bias/v
&:$	2Adam/dense_1/kernel/v
:2Adam/dense_1/bias/v
&:$	2Adam/dense_2/kernel/v
:2Adam/dense_2/bias/v
&:$	2Adam/dense_3/kernel/v
:2Adam/dense_3/bias/v
%:#2Adam/output1/kernel/v
:2Adam/output1/bias/v
%:#2Adam/output2/kernel/v
:2Adam/output2/bias/v
%:#2Adam/output3/kernel/v
:2Adam/output3/bias/v
%:#2Adam/output4/kernel/v
:2Adam/output4/bias/v
æ2ã
&__inference_model_layer_call_fn_101103
&__inference_model_layer_call_fn_100322
&__inference_model_layer_call_fn_101220
&__inference_model_layer_call_fn_100041À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
è2å
 __inference__wrapped_model_97997À
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *0¢-
+(
input_1ÿÿÿÿÿÿÿÿÿ¬¬
Ð2Í
A__inference_model_layer_call_and_return_conditional_losses_100986
A__inference_model_layer_call_and_return_conditional_losses_100725
@__inference_model_layer_call_and_return_conditional_losses_99595
@__inference_model_layer_call_and_return_conditional_losses_99759À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
-__inference_stem_conv_pad_layer_call_fn_98010à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
°2­
H__inference_stem_conv_pad_layer_call_and_return_conditional_losses_98004à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ô2Ñ
*__inference_stem_conv_layer_call_fn_101234¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_stem_conv_layer_call_and_return_conditional_losses_101227¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
â2ß
(__inference_stem_bn_layer_call_fn_101358
(__inference_stem_bn_layer_call_fn_101296
(__inference_stem_bn_layer_call_fn_101283
(__inference_stem_bn_layer_call_fn_101345´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Î2Ë
C__inference_stem_bn_layer_call_and_return_conditional_losses_101332
C__inference_stem_bn_layer_call_and_return_conditional_losses_101252
C__inference_stem_bn_layer_call_and_return_conditional_losses_101270
C__inference_stem_bn_layer_call_and_return_conditional_losses_101314´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ú2×
0__inference_stem_activation_layer_call_fn_101373¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_stem_activation_layer_call_and_return_conditional_losses_101368¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
.__inference_block1a_dwconv_layer_call_fn_98128×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(
¨2¥
I__inference_block1a_dwconv_layer_call_and_return_conditional_losses_98120×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(
î2ë
+__inference_block1a_bn_layer_call_fn_101435
+__inference_block1a_bn_layer_call_fn_101484
+__inference_block1a_bn_layer_call_fn_101422
+__inference_block1a_bn_layer_call_fn_101497´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ú2×
F__inference_block1a_bn_layer_call_and_return_conditional_losses_101391
F__inference_block1a_bn_layer_call_and_return_conditional_losses_101471
F__inference_block1a_bn_layer_call_and_return_conditional_losses_101409
F__inference_block1a_bn_layer_call_and_return_conditional_losses_101453´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ý2Ú
3__inference_block1a_activation_layer_call_fn_101512¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ø2õ
N__inference_block1a_activation_layer_call_and_return_conditional_losses_101507¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
2__inference_block1a_se_squeeze_layer_call_fn_98241à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
µ2²
M__inference_block1a_se_squeeze_layer_call_and_return_conditional_losses_98235à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ý2Ú
3__inference_block1a_se_reshape_layer_call_fn_101531¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ø2õ
N__inference_block1a_se_reshape_layer_call_and_return_conditional_losses_101526¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ü2Ù
2__inference_block1a_se_reduce_layer_call_fn_101556¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷2ô
M__inference_block1a_se_reduce_layer_call_and_return_conditional_losses_101547¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ü2Ù
2__inference_block1a_se_expand_layer_call_fn_101576¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷2ô
M__inference_block1a_se_expand_layer_call_and_return_conditional_losses_101567¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ü2Ù
2__inference_block1a_se_excite_layer_call_fn_101588¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷2ô
M__inference_block1a_se_excite_layer_call_and_return_conditional_losses_101582¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_block1a_project_conv_layer_call_fn_101602¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_block1a_project_conv_layer_call_and_return_conditional_losses_101595¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
3__inference_block1a_project_bn_layer_call_fn_101651
3__inference_block1a_project_bn_layer_call_fn_101726
3__inference_block1a_project_bn_layer_call_fn_101713
3__inference_block1a_project_bn_layer_call_fn_101664´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ú2÷
N__inference_block1a_project_bn_layer_call_and_return_conditional_losses_101620
N__inference_block1a_project_bn_layer_call_and_return_conditional_losses_101700
N__inference_block1a_project_bn_layer_call_and_return_conditional_losses_101638
N__inference_block1a_project_bn_layer_call_and_return_conditional_losses_101682´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
.__inference_block1b_dwconv_layer_call_fn_98359×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¨2¥
I__inference_block1b_dwconv_layer_call_and_return_conditional_losses_98351×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
î2ë
+__inference_block1b_bn_layer_call_fn_101837
+__inference_block1b_bn_layer_call_fn_101788
+__inference_block1b_bn_layer_call_fn_101850
+__inference_block1b_bn_layer_call_fn_101775´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ú2×
F__inference_block1b_bn_layer_call_and_return_conditional_losses_101762
F__inference_block1b_bn_layer_call_and_return_conditional_losses_101824
F__inference_block1b_bn_layer_call_and_return_conditional_losses_101744
F__inference_block1b_bn_layer_call_and_return_conditional_losses_101806´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ý2Ú
3__inference_block1b_activation_layer_call_fn_101865¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ø2õ
N__inference_block1b_activation_layer_call_and_return_conditional_losses_101860¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
2__inference_block1b_se_squeeze_layer_call_fn_98472à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
µ2²
M__inference_block1b_se_squeeze_layer_call_and_return_conditional_losses_98466à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ý2Ú
3__inference_block1b_se_reshape_layer_call_fn_101884¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ø2õ
N__inference_block1b_se_reshape_layer_call_and_return_conditional_losses_101879¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ü2Ù
2__inference_block1b_se_reduce_layer_call_fn_101909¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷2ô
M__inference_block1b_se_reduce_layer_call_and_return_conditional_losses_101900¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ü2Ù
2__inference_block1b_se_expand_layer_call_fn_101929¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷2ô
M__inference_block1b_se_expand_layer_call_and_return_conditional_losses_101920¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ü2Ù
2__inference_block1b_se_excite_layer_call_fn_101941¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷2ô
M__inference_block1b_se_excite_layer_call_and_return_conditional_losses_101935¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_block1b_project_conv_layer_call_fn_101955¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_block1b_project_conv_layer_call_and_return_conditional_losses_101948¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
3__inference_block1b_project_bn_layer_call_fn_102004
3__inference_block1b_project_bn_layer_call_fn_102017
3__inference_block1b_project_bn_layer_call_fn_102079
3__inference_block1b_project_bn_layer_call_fn_102066´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ú2÷
N__inference_block1b_project_bn_layer_call_and_return_conditional_losses_101973
N__inference_block1b_project_bn_layer_call_and_return_conditional_losses_102035
N__inference_block1b_project_bn_layer_call_and_return_conditional_losses_101991
N__inference_block1b_project_bn_layer_call_and_return_conditional_losses_102053´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
-__inference_block1b_drop_layer_call_fn_102114
-__inference_block1b_drop_layer_call_fn_102109´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Î2Ë
H__inference_block1b_drop_layer_call_and_return_conditional_losses_102099
H__inference_block1b_drop_layer_call_and_return_conditional_losses_102104´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ö2Ó
,__inference_block1b_add_layer_call_fn_102126¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_block1b_add_layer_call_and_return_conditional_losses_102120¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
1__inference_average_pooling2d_layer_call_fn_98584à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
´2±
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_98578à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ñ2Î
'__inference_conv2d_layer_call_fn_102145¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_conv2d_layer_call_and_return_conditional_losses_102136¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
-__inference_max_pooling2d_layer_call_fn_98596à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
°2­
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_98590à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ò2Ï
(__inference_flatten_layer_call_fn_102156¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_flatten_layer_call_and_return_conditional_losses_102151¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ð2Í
&__inference_dense_layer_call_fn_102176¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ë2è
A__inference_dense_layer_call_and_return_conditional_losses_102167¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense_1_layer_call_fn_102196¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_1_layer_call_and_return_conditional_losses_102187¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense_2_layer_call_fn_102216¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_2_layer_call_and_return_conditional_losses_102207¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense_3_layer_call_fn_102236¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_3_layer_call_and_return_conditional_losses_102227¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_output1_layer_call_fn_102255¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_output1_layer_call_and_return_conditional_losses_102246¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_output2_layer_call_fn_102274¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_output2_layer_call_and_return_conditional_losses_102265¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_output3_layer_call_fn_102293¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_output3_layer_call_and_return_conditional_losses_102284¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_output4_layer_call_fn_102312¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_output4_layer_call_and_return_conditional_losses_102303¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ËBÈ
$__inference_signature_wrapper_100449input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ù
 __inference__wrapped_model_97997ÔW45<BCDENTUVWhinox~ ¡¦§°¶·¸¹ÊËêëäåÞßØÙüýö÷ðñ:¢7
0¢-
+(
input_1ÿÿÿÿÿÿÿÿÿ¬¬
ª "¼ª¸
,
output1!
output1ÿÿÿÿÿÿÿÿÿ
,
output2!
output2ÿÿÿÿÿÿÿÿÿ
,
output3!
output3ÿÿÿÿÿÿÿÿÿ
,
output4!
output4ÿÿÿÿÿÿÿÿÿï
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_98578R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ç
1__inference_average_pooling2d_layer_call_fn_98584R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¾
N__inference_block1a_activation_layer_call_and_return_conditional_losses_101507l9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ(
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ(
 
3__inference_block1a_activation_layer_call_fn_101512_9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ(
ª ""ÿÿÿÿÿÿÿÿÿ(á
F__inference_block1a_bn_layer_call_and_return_conditional_losses_101391TUVWM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(
 á
F__inference_block1a_bn_layer_call_and_return_conditional_losses_101409TUVWM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(
 À
F__inference_block1a_bn_layer_call_and_return_conditional_losses_101453vTUVW=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ(
p
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ(
 À
F__inference_block1a_bn_layer_call_and_return_conditional_losses_101471vTUVW=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ(
p 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ(
 ¹
+__inference_block1a_bn_layer_call_fn_101422TUVWM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(¹
+__inference_block1a_bn_layer_call_fn_101435TUVWM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(
+__inference_block1a_bn_layer_call_fn_101484iTUVW=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ(
p
ª ""ÿÿÿÿÿÿÿÿÿ(
+__inference_block1a_bn_layer_call_fn_101497iTUVW=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ(
p 
ª ""ÿÿÿÿÿÿÿÿÿ(Ý
I__inference_block1a_dwconv_layer_call_and_return_conditional_losses_98120NI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(
 µ
.__inference_block1a_dwconv_layer_call_fn_98128NI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(ë
N__inference_block1a_project_bn_layer_call_and_return_conditional_losses_101620~M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ë
N__inference_block1a_project_bn_layer_call_and_return_conditional_losses_101638~M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ê
N__inference_block1a_project_bn_layer_call_and_return_conditional_losses_101682x~=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 Ê
N__inference_block1a_project_bn_layer_call_and_return_conditional_losses_101700x~=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 Ã
3__inference_block1a_project_bn_layer_call_fn_101651~M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÃ
3__inference_block1a_project_bn_layer_call_fn_101664~M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
3__inference_block1a_project_bn_layer_call_fn_101713k~=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p
ª ""ÿÿÿÿÿÿÿÿÿ¢
3__inference_block1a_project_bn_layer_call_fn_101726k~=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ""ÿÿÿÿÿÿÿÿÿÃ
P__inference_block1a_project_conv_layer_call_and_return_conditional_losses_101595ox9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ(
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 
5__inference_block1a_project_conv_layer_call_fn_101602bx9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ(
ª ""ÿÿÿÿÿÿÿÿÿñ
M__inference_block1a_se_excite_layer_call_and_return_conditional_losses_101582l¢i
b¢_
]Z
,)
inputs/0ÿÿÿÿÿÿÿÿÿ(
*'
inputs/1ÿÿÿÿÿÿÿÿÿ(
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ(
 É
2__inference_block1a_se_excite_layer_call_fn_101588l¢i
b¢_
]Z
,)
inputs/0ÿÿÿÿÿÿÿÿÿ(
*'
inputs/1ÿÿÿÿÿÿÿÿÿ(
ª ""ÿÿÿÿÿÿÿÿÿ(½
M__inference_block1a_se_expand_layer_call_and_return_conditional_losses_101567lno7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ

ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ(
 
2__inference_block1a_se_expand_layer_call_fn_101576_no7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ

ª " ÿÿÿÿÿÿÿÿÿ(½
M__inference_block1a_se_reduce_layer_call_and_return_conditional_losses_101547lhi7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ(
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ

 
2__inference_block1a_se_reduce_layer_call_fn_101556_hi7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ(
ª " ÿÿÿÿÿÿÿÿÿ
²
N__inference_block1a_se_reshape_layer_call_and_return_conditional_losses_101526`/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ(
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ(
 
3__inference_block1a_se_reshape_layer_call_fn_101531S/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ(
ª " ÿÿÿÿÿÿÿÿÿ(Ö
M__inference_block1a_se_squeeze_layer_call_and_return_conditional_losses_98235R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ­
2__inference_block1a_se_squeeze_layer_call_fn_98241wR¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¾
N__inference_block1b_activation_layer_call_and_return_conditional_losses_101860l9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 
3__inference_block1b_activation_layer_call_fn_101865_9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ
ª ""ÿÿÿÿÿÿÿÿÿí
G__inference_block1b_add_layer_call_and_return_conditional_losses_102120¡n¢k
d¢a
_\
,)
inputs/0ÿÿÿÿÿÿÿÿÿ
,)
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 Å
,__inference_block1b_add_layer_call_fn_102126n¢k
d¢a
_\
,)
inputs/0ÿÿÿÿÿÿÿÿÿ
,)
inputs/1ÿÿÿÿÿÿÿÿÿ
ª ""ÿÿÿÿÿÿÿÿÿÄ
F__inference_block1b_bn_layer_call_and_return_conditional_losses_101744z=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 Ä
F__inference_block1b_bn_layer_call_and_return_conditional_losses_101762z=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 å
F__inference_block1b_bn_layer_call_and_return_conditional_losses_101806M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 å
F__inference_block1b_bn_layer_call_and_return_conditional_losses_101824M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
+__inference_block1b_bn_layer_call_fn_101775m=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p
ª ""ÿÿÿÿÿÿÿÿÿ
+__inference_block1b_bn_layer_call_fn_101788m=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ""ÿÿÿÿÿÿÿÿÿ½
+__inference_block1b_bn_layer_call_fn_101837M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ½
+__inference_block1b_bn_layer_call_fn_101850M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¼
H__inference_block1b_drop_layer_call_and_return_conditional_losses_102099p=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 ¼
H__inference_block1b_drop_layer_call_and_return_conditional_losses_102104p=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 
-__inference_block1b_drop_layer_call_fn_102109c=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p
ª ""ÿÿÿÿÿÿÿÿÿ
-__inference_block1b_drop_layer_call_fn_102114c=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ""ÿÿÿÿÿÿÿÿÿÞ
I__inference_block1b_dwconv_layer_call_and_return_conditional_losses_98351I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¶
.__inference_block1b_dwconv_layer_call_fn_98359I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿí
N__inference_block1b_project_bn_layer_call_and_return_conditional_losses_101973¶·¸¹M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 í
N__inference_block1b_project_bn_layer_call_and_return_conditional_losses_101991¶·¸¹M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ì
N__inference_block1b_project_bn_layer_call_and_return_conditional_losses_102035z¶·¸¹=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 Ì
N__inference_block1b_project_bn_layer_call_and_return_conditional_losses_102053z¶·¸¹=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 Å
3__inference_block1b_project_bn_layer_call_fn_102004¶·¸¹M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÅ
3__inference_block1b_project_bn_layer_call_fn_102017¶·¸¹M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¤
3__inference_block1b_project_bn_layer_call_fn_102066m¶·¸¹=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p
ª ""ÿÿÿÿÿÿÿÿÿ¤
3__inference_block1b_project_bn_layer_call_fn_102079m¶·¸¹=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ""ÿÿÿÿÿÿÿÿÿÄ
P__inference_block1b_project_conv_layer_call_and_return_conditional_losses_101948p°9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 
5__inference_block1b_project_conv_layer_call_fn_101955c°9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ
ª ""ÿÿÿÿÿÿÿÿÿñ
M__inference_block1b_se_excite_layer_call_and_return_conditional_losses_101935l¢i
b¢_
]Z
,)
inputs/0ÿÿÿÿÿÿÿÿÿ
*'
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 É
2__inference_block1b_se_excite_layer_call_fn_101941l¢i
b¢_
]Z
,)
inputs/0ÿÿÿÿÿÿÿÿÿ
*'
inputs/1ÿÿÿÿÿÿÿÿÿ
ª ""ÿÿÿÿÿÿÿÿÿ¿
M__inference_block1b_se_expand_layer_call_and_return_conditional_losses_101920n¦§7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
2__inference_block1b_se_expand_layer_call_fn_101929a¦§7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿ¿
M__inference_block1b_se_reduce_layer_call_and_return_conditional_losses_101900n ¡7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
2__inference_block1b_se_reduce_layer_call_fn_101909a ¡7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿ²
N__inference_block1b_se_reshape_layer_call_and_return_conditional_losses_101879`/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
3__inference_block1b_se_reshape_layer_call_fn_101884S/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿÖ
M__inference_block1b_se_squeeze_layer_call_and_return_conditional_losses_98466R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ­
2__inference_block1b_se_squeeze_layer_call_fn_98472wR¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´
B__inference_conv2d_layer_call_and_return_conditional_losses_102136nÊË7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
'__inference_conv2d_layer_call_fn_102145aÊË7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿ¦
C__inference_dense_1_layer_call_and_return_conditional_losses_102187_Þß0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
(__inference_dense_1_layer_call_fn_102196RÞß0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
C__inference_dense_2_layer_call_and_return_conditional_losses_102207_äå0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
(__inference_dense_2_layer_call_fn_102216Räå0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
C__inference_dense_3_layer_call_and_return_conditional_losses_102227_êë0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
(__inference_dense_3_layer_call_fn_102236Rêë0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¤
A__inference_dense_layer_call_and_return_conditional_losses_102167_ØÙ0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
&__inference_dense_layer_call_fn_102176RØÙ0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¨
C__inference_flatten_layer_call_and_return_conditional_losses_102151a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
(__inference_flatten_layer_call_fn_102156T7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿë
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_98590R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ã
-__inference_max_pooling2d_layer_call_fn_98596R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿï
A__inference_model_layer_call_and_return_conditional_losses_100725©W45<BCDENTUVWhinox~ ¡¦§°¶·¸¹ÊËêëäåÞßØÙüýö÷ðñA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ¬¬
p

 
ª "¢
|

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ

0/2ÿÿÿÿÿÿÿÿÿ

0/3ÿÿÿÿÿÿÿÿÿ
 ï
A__inference_model_layer_call_and_return_conditional_losses_100986©W45<BCDENTUVWhinox~ ¡¦§°¶·¸¹ÊËêëäåÞßØÙüýö÷ðñA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ¬¬
p 

 
ª "¢
|

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ

0/2ÿÿÿÿÿÿÿÿÿ

0/3ÿÿÿÿÿÿÿÿÿ
 ï
@__inference_model_layer_call_and_return_conditional_losses_99595ªW45<BCDENTUVWhinox~ ¡¦§°¶·¸¹ÊËêëäåÞßØÙüýö÷ðñB¢?
8¢5
+(
input_1ÿÿÿÿÿÿÿÿÿ¬¬
p

 
ª "¢
|

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ

0/2ÿÿÿÿÿÿÿÿÿ

0/3ÿÿÿÿÿÿÿÿÿ
 ï
@__inference_model_layer_call_and_return_conditional_losses_99759ªW45<BCDENTUVWhinox~ ¡¦§°¶·¸¹ÊËêëäåÞßØÙüýö÷ðñB¢?
8¢5
+(
input_1ÿÿÿÿÿÿÿÿÿ¬¬
p 

 
ª "¢
|

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ

0/2ÿÿÿÿÿÿÿÿÿ

0/3ÿÿÿÿÿÿÿÿÿ
 Á
&__inference_model_layer_call_fn_100041W45<BCDENTUVWhinox~ ¡¦§°¶·¸¹ÊËêëäåÞßØÙüýö÷ðñB¢?
8¢5
+(
input_1ÿÿÿÿÿÿÿÿÿ¬¬
p

 
ª "wt

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ

2ÿÿÿÿÿÿÿÿÿ

3ÿÿÿÿÿÿÿÿÿÁ
&__inference_model_layer_call_fn_100322W45<BCDENTUVWhinox~ ¡¦§°¶·¸¹ÊËêëäåÞßØÙüýö÷ðñB¢?
8¢5
+(
input_1ÿÿÿÿÿÿÿÿÿ¬¬
p 

 
ª "wt

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ

2ÿÿÿÿÿÿÿÿÿ

3ÿÿÿÿÿÿÿÿÿÀ
&__inference_model_layer_call_fn_101103W45<BCDENTUVWhinox~ ¡¦§°¶·¸¹ÊËêëäåÞßØÙüýö÷ðñA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ¬¬
p

 
ª "wt

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ

2ÿÿÿÿÿÿÿÿÿ

3ÿÿÿÿÿÿÿÿÿÀ
&__inference_model_layer_call_fn_101220W45<BCDENTUVWhinox~ ¡¦§°¶·¸¹ÊËêëäåÞßØÙüýö÷ðñA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ¬¬
p 

 
ª "wt

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ

2ÿÿÿÿÿÿÿÿÿ

3ÿÿÿÿÿÿÿÿÿ¥
C__inference_output1_layer_call_and_return_conditional_losses_102246^ðñ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
(__inference_output1_layer_call_fn_102255Qðñ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¥
C__inference_output2_layer_call_and_return_conditional_losses_102265^ö÷/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
(__inference_output2_layer_call_fn_102274Qö÷/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¥
C__inference_output3_layer_call_and_return_conditional_losses_102284^üý/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
(__inference_output3_layer_call_fn_102293Qüý/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¥
C__inference_output4_layer_call_and_return_conditional_losses_102303^/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
(__inference_output4_layer_call_fn_102312Q/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
$__inference_signature_wrapper_100449ßW45<BCDENTUVWhinox~ ¡¦§°¶·¸¹ÊËêëäåÞßØÙüýö÷ðñE¢B
¢ 
;ª8
6
input_1+(
input_1ÿÿÿÿÿÿÿÿÿ¬¬"¼ª¸
,
output1!
output1ÿÿÿÿÿÿÿÿÿ
,
output2!
output2ÿÿÿÿÿÿÿÿÿ
,
output3!
output3ÿÿÿÿÿÿÿÿÿ
,
output4!
output4ÿÿÿÿÿÿÿÿÿ»
K__inference_stem_activation_layer_call_and_return_conditional_losses_101368l9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ(
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ(
 
0__inference_stem_activation_layer_call_fn_101373_9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ(
ª ""ÿÿÿÿÿÿÿÿÿ(Þ
C__inference_stem_bn_layer_call_and_return_conditional_losses_101252BCDEM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(
 Þ
C__inference_stem_bn_layer_call_and_return_conditional_losses_101270BCDEM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(
 ½
C__inference_stem_bn_layer_call_and_return_conditional_losses_101314vBCDE=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ(
p
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ(
 ½
C__inference_stem_bn_layer_call_and_return_conditional_losses_101332vBCDE=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ(
p 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ(
 ¶
(__inference_stem_bn_layer_call_fn_101283BCDEM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(¶
(__inference_stem_bn_layer_call_fn_101296BCDEM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(
(__inference_stem_bn_layer_call_fn_101345iBCDE=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ(
p
ª ""ÿÿÿÿÿÿÿÿÿ(
(__inference_stem_bn_layer_call_fn_101358iBCDE=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ(
p 
ª ""ÿÿÿÿÿÿÿÿÿ(¸
E__inference_stem_conv_layer_call_and_return_conditional_losses_101227o<9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ­­
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ(
 
*__inference_stem_conv_layer_call_fn_101234b<9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ­­
ª ""ÿÿÿÿÿÿÿÿÿ(ë
H__inference_stem_conv_pad_layer_call_and_return_conditional_losses_98004R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ã
-__inference_stem_conv_pad_layer_call_fn_98010R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ