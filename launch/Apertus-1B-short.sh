# Recommended nodes: 1
# ETA: ~8 x 1x4GPUh
# Throughput w/4GPU: ~42260
# FLOPs/iter: ~18.703e15
# Total FLOPs: ~46.757e18
source .env
bash launch/submit.sh 1 \
	--iters 2500 \
	$*
