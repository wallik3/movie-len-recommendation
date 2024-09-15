from movie_len_recommendation.architecture.mfnet import (
    MFNet, 
    MFNetSigmoidRange
)

def get_mf_architecture(version="default"):
  """
    Every MFNet require the same argument <n_users, n_items, emb_size>
  """
  mf_architecture_map = {
      "default" : MFNet,
      "with_sigmoid_range" : MFNetSigmoidRange,
  }
  mf_architecture = mf_architecture_map.get(version,mf_architecture_map["default"])
  return mf_architecture