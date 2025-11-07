# TE ETP (MBS3): 23.3k tsg
# TE causal (MBS3): 19.7k
# TE arbitrary (MBS1): 14.5k


# TE arbitrary naive-compiled (MBS1): 15.2k
# TE causal naive-compiled (MBS3): 21.3k
# TE arbitrary naive-compiled (MBS3 nodes2): 

# flex _compile=False (MBS2): 5.1k
# flex (MBS2): 5.1k
# flex bs=256 naive-compiled (MBS2): 5.2k


# TE arbitrary naive-compiled (MBS2): OOM

# Recommended nodes: 2

source .env
export MBS=1
bash launch/submit.sh 1 \
	--iters 2500 \
	--n-recurrences 5 \
	--n-encode 7 \
	--n-think 4 \
	--n-decode 5 \
	--latent-masker topk \
	--latent-topk-masker-k 128 \
	$*
