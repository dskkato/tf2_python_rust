
,
xPlaceholder*
dtype0*
shape:
,
yPlaceholder*
dtype0*
shape:
B
random_uniform/shapeConst*
dtype0*
valueB:
?
random_uniform/minConst*
dtype0*
valueB
 *  ??
?
random_uniform/maxConst*
dtype0*
valueB
 *  ??
r
random_uniform/RandomUniformRandomUniformrandom_uniform/shape*
T0*
dtype0*

seed *
seed2 
J
random_uniform/subSubrandom_uniform/maxrandom_uniform/min*
T0
T
random_uniform/mulMulrandom_uniform/RandomUniformrandom_uniform/sub*
T0
F
random_uniformAddrandom_uniform/mulrandom_uniform/min*
T0
?
wVarHandleOp*
_class

loc:@w*
allowed_devices
 *
	container *
dtype0*
shape:*
shared_namew
;
"w/IsInitialized/VarIsInitializedOpVarIsInitializedOpw
<
w/AssignAssignVariableOpwrandom_uniform*
dtype0
7
w/Read/ReadVariableOpReadVariableOpw*
dtype0
6
zerosConst*
dtype0*
valueB*    
?
bVarHandleOp*
_class

loc:@b*
allowed_devices
 *
	container *
dtype0*
shape:*
shared_nameb
;
"b/IsInitialized/VarIsInitializedOpVarIsInitializedOpb
3
b/AssignAssignVariableOpbzeros*
dtype0
7
b/Read/ReadVariableOpReadVariableOpb*
dtype0
0
ReadVariableOpReadVariableOpw*
dtype0
&
mulMulReadVariableOpx*
T0
4
add/ReadVariableOpReadVariableOpb*
dtype0
.
addAddV2muladd/ReadVariableOp*
T0

subSubaddy*
T0

SquareSquaresub*
T0

RankRankSquare*
T0
5
range/startConst*
dtype0*
value	B : 
5
range/deltaConst*
dtype0*
value	B :
:
rangeRangerange/startRankrange/delta*

Tidx0
A
MeanMeanSquarerange*
T0*

Tidx0*
	keep_dims( 
8
gradients/ShapeConst*
dtype0*
valueB 
F
gradients/grad_ys_0/ConstConst*
dtype0*
valueB
 *  ??
b
gradients/grad_ys_0Fillgradients/Shapegradients/grad_ys_0/Const*
T0*

index_type0
C
gradients/Mean_grad/ShapeShapeSquare*
T0*
out_type0
?
gradients/Mean_grad/SizeSizegradients/Mean_grad/Shape*
T0*,
_class"
 loc:@gradients/Mean_grad/Shape*
out_type0
x
gradients/Mean_grad/addAddV2rangegradients/Mean_grad/Size*
T0*,
_class"
 loc:@gradients/Mean_grad/Shape
?
gradients/Mean_grad/modFloorModgradients/Mean_grad/addgradients/Mean_grad/Size*
T0*,
_class"
 loc:@gradients/Mean_grad/Shape
?
gradients/Mean_grad/Shape_1Shapegradients/Mean_grad/mod*
T0*,
_class"
 loc:@gradients/Mean_grad/Shape*
out_type0
w
gradients/Mean_grad/range/startConst*,
_class"
 loc:@gradients/Mean_grad/Shape*
dtype0*
value	B : 
w
gradients/Mean_grad/range/deltaConst*,
_class"
 loc:@gradients/Mean_grad/Shape*
dtype0*
value	B :
?
gradients/Mean_grad/rangeRangegradients/Mean_grad/range/startgradients/Mean_grad/Sizegradients/Mean_grad/range/delta*

Tidx0*,
_class"
 loc:@gradients/Mean_grad/Shape
v
gradients/Mean_grad/ones/ConstConst*,
_class"
 loc:@gradients/Mean_grad/Shape*
dtype0*
value	B :
?
gradients/Mean_grad/onesFillgradients/Mean_grad/Shape_1gradients/Mean_grad/ones/Const*
T0*,
_class"
 loc:@gradients/Mean_grad/Shape*

index_type0
?
!gradients/Mean_grad/DynamicStitchDynamicStitchgradients/Mean_grad/rangegradients/Mean_grad/modgradients/Mean_grad/Shapegradients/Mean_grad/ones*
N*
T0*,
_class"
 loc:@gradients/Mean_grad/Shape
u
gradients/Mean_grad/ReshapeReshapegradients/grad_ys_0!gradients/Mean_grad/DynamicStitch*
T0*
Tshape0
{
gradients/Mean_grad/BroadcastToBroadcastTogradients/Mean_grad/Reshapegradients/Mean_grad/Shape*
T0*

Tidx0
E
gradients/Mean_grad/Shape_2ShapeSquare*
T0*
out_type0
D
gradients/Mean_grad/Shape_3Const*
dtype0*
valueB 
G
gradients/Mean_grad/ConstConst*
dtype0*
valueB: 
~
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_2gradients/Mean_grad/Const*
T0*

Tidx0*
	keep_dims( 
I
gradients/Mean_grad/Const_1Const*
dtype0*
valueB: 
?
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_3gradients/Mean_grad/Const_1*
T0*

Tidx0*
	keep_dims( 
G
gradients/Mean_grad/Maximum/yConst*
dtype0*
value	B :
j
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0
h
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0
f
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

DstT0*

SrcT0*
Truncate( 
j
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/BroadcastTogradients/Mean_grad/Cast*
T0
f
gradients/Square_grad/ConstConst^gradients/Mean_grad/truediv*
dtype0*
valueB
 *   @
K
gradients/Square_grad/MulMulsubgradients/Square_grad/Const*
T0
c
gradients/Square_grad/Mul_1Mulgradients/Mean_grad/truedivgradients/Square_grad/Mul*
T0
?
gradients/sub_grad/ShapeShapeadd*
T0*
out_type0
?
gradients/sub_grad/Shape_1Shapey*
T0*
out_type0
?
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*
T0
?
gradients/sub_grad/SumSumgradients/Square_grad/Mul_1(gradients/sub_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
n
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
T0*
Tshape0
C
gradients/sub_grad/NegNeggradients/Square_grad/Mul_1*
T0
?
gradients/sub_grad/Sum_1Sumgradients/sub_grad/Neg*gradients/sub_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
t
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Sum_1gradients/sub_grad/Shape_1*
T0*
Tshape0
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
?
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape
?
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1
?
gradients/add_grad/ShapeShapemul*
T0*
out_type0
P
gradients/add_grad/Shape_1Shapeadd/ReadVariableOp*
T0*
out_type0
?
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0
?
gradients/add_grad/SumSum+gradients/sub_grad/tuple/control_dependency(gradients/add_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
n
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0
?
gradients/add_grad/Sum_1Sum+gradients/sub_grad/tuple/control_dependency*gradients/add_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
t
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
T0*
Tshape0
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
?
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/add_grad/Reshape
?
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1
J
gradients/mul_grad/ShapeShapeReadVariableOp*
T0*
out_type0
?
gradients/mul_grad/Shape_1Shapex*
T0*
out_type0
?
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*
T0
V
gradients/mul_grad/MulMul+gradients/add_grad/tuple/control_dependencyx*
T0
?
gradients/mul_grad/SumSumgradients/mul_grad/Mul(gradients/mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
n
gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*
T0*
Tshape0
e
gradients/mul_grad/Mul_1MulReadVariableOp+gradients/add_grad/tuple/control_dependency*
T0
?
gradients/mul_grad/Sum_1Sumgradients/mul_grad/Mul_1*gradients/mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
t
gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*
T0*
Tshape0
g
#gradients/mul_grad/tuple/group_depsNoOp^gradients/mul_grad/Reshape^gradients/mul_grad/Reshape_1
?
+gradients/mul_grad/tuple/control_dependencyIdentitygradients/mul_grad/Reshape$^gradients/mul_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/mul_grad/Reshape
?
-gradients/mul_grad/tuple/control_dependency_1Identitygradients/mul_grad/Reshape_1$^gradients/mul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_grad/Reshape_1
@
train/learning_rateConst*
dtype0*
valueB
 *   ?
?
+train/update_w/ResourceApplyGradientDescentResourceApplyGradientDescentwtrain/learning_rate+gradients/mul_grad/tuple/control_dependency*
T0*
_class

loc:@w*
use_locking( 
?
+train/update_b/ResourceApplyGradientDescentResourceApplyGradientDescentbtrain/learning_rate-gradients/add_grad/tuple/control_dependency_1*
T0*
_class

loc:@b*
use_locking( 
i
trainNoOp,^train/update_b/ResourceApplyGradientDescent,^train/update_w/ResourceApplyGradientDescent
"
initNoOp	^b/Assign	^w/Assign
A
save/filename/inputConst*
dtype0*
valueB Bmodel
V
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
shape: 
M

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
shape: 
I
save/SaveV2/tensor_namesConst*
dtype0*
valueBBbBw
K
save/SaveV2/shape_and_slicesConst*
dtype0*
valueBB B 
?
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesb/Read/ReadVariableOpw/Read/ReadVariableOp*
dtypes
2
e
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const
[
save/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*
valueBBbBw
]
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
valueBB B 
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
2
save/IdentityIdentitysave/RestoreV2*
T0
H
save/AssignVariableOpAssignVariableOpbsave/Identity*
dtype0
6
save/Identity_1Identitysave/RestoreV2:1*
T0
L
save/AssignVariableOp_1AssignVariableOpwsave/Identity_1*
dtype0
J
save/restore_allNoOp^save/AssignVariableOp^save/AssignVariableOp_1"?