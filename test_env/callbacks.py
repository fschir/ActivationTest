from operator import attrgetter


class Callback:
    order = 0


class TestCallback(Callback):
    def __init__(self, hook_step):
        self.name = hook_step

    def __call__(self, f):
        def _f(o, *args, **kwargs):
            try:
                o.callback(f'before_{self.name}')
                f(o, *args, **kwargs)
                o.callback(f'after_{self.name}')
            except globals()[f'Broken cb {self.name.title()}']:
                pass
            finally:
                o.callback(f'cleanup_{self.name}')
        return _f
    

class CountingCallback(Callback):
    def __init__(self):
        self.count = None

    def before_fit(self): 
        self.count = 0
        
    def after_batch(self):
        self.count += 1 
    
    def after_fit(self):
        print(f'Done {self.count} batches')


class MeasureActivationsCallback(Callback):
    def after_backward(self):
        for layer in self.model.layers:
            pass 


def run_cbs(cbs, method_name, caller=None):
    for cb in sorted(cbs, key=attrgetter('order')):
        method = getattr(cb, method_name, None)
        if method is not None:
            method()
