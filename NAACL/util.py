SECS_PER_MIN = 60
SECS_PER_HOUR = SECS_PER_MIN * 60
SECS_PER_DAY = SECS_PER_HOUR * 24

def secs_to_str(secs):
    days = int(secs) // SECS_PER_DAY
    secs -= days * SECS_PER_DAY
    hours = int(secs) // SECS_PER_HOUR
    secs -= hours * SECS_PER_HOUR
    mins = int(secs) // SECS_PER_MIN
    secs -= mins * SECS_PER_MIN
    if days > 0:
        return '%dd%02dh%02dm' % (days, hours, mins)
    elif hours > 0:
        return '%dh%02dm%02ds' % (hours, mins, int(secs))
    elif mins > 0:
        return '%dm%02ds' % (mins, int(secs))
    elif secs >= 1:
        return '%.1fs' % secs
    return '%.2fs' % secs

def get_prf(tp, fp, fn, get_str=False):
  """Get precision, recall, f1 from true pos, false pos, false neg."""
  if tp + fp == 0:
    precision = 0
  else:
    precision = float(tp) / (tp + fp)
  if tp + fn == 0:
    recall = 0
  else:
    recall = float(tp) / (tp + fn)
  if precision + recall == 0:
    f1 = 0
  else:
    f1 = 2 * precision * recall / (precision + recall)
  if get_str:
    return '\n'.join([
        'Precision: %.2f%%' % (100.0 * precision),
        'Recall   : %.2f%%' % (100.0 * recall),
        'F1       : %.2f%%' % (100.0 * f1)])
  return precision, recall, f1