from esinet import Net
from esinet.forward import create_forward_model, get_info
import mne

info = get_info(sfreq=100)
fwd = create_forward_model(info=info, sampling='ico3')
net = Net(fwd, model_type='convdip')
net._build_model()
print(hasattr(net, 'model'))
