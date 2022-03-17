from torch.utils.tensorboard.writer import SummaryWriter, hparams
from ice.llutil.argparser import args as ice_args


class BoardWriter(SummaryWriter):
  
    def __init__(self, log_dir=None, comment='', purge_step=None, max_queue=10, flush_secs=120, filename_suffix=''):
        super().__init__(log_dir, comment, purge_step, max_queue, flush_secs, filename_suffix)
        self._scalar_names = set()
        
    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False, double_precision=False):
        self._scalar_names.add(tag)
        return super().add_scalar(tag, scalar_value, global_step, walltime, new_style, double_precision)
    
    def add_hparams(
        self, hparam_domain_discrete=None
    ):
        """Add a set of hyperparameters to be compared in TensorBoard.

        Args:
            hparam_domain_discrete: (Optional[Dict[str, List[Any]]]) A dictionary that
              contains names of the hyperparameters and all discrete values they can hold

        """
        # torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        hparam_dict = ice_args.hparam_dict()
        if len(hparam_dict) == 0: return
        
        metric_dict = {k:0. for k in self._scalar_names}
        exp, ssi, sei = hparams(hparam_dict, metric_dict, hparam_domain_discrete)

        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)