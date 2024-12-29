class AlgoSpecs:
  def __init__(self, **kwargs):
    self.td_errors = []

  def policy(self, state):
    pass

  def reset(self):
    pass

  def update(self, **kwargs):
    pass

  def predict(self, state):
    pass

  def save(self, filename):
    pass

  def load(self, filename):
    pass


class SearchCV:
  def new(self, **params):
    pass

  def search(self, episodes, samples):
    pass

  def summary(self):
    pass

  def plot_metrics(self, index):
    pass
