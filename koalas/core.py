"""
Koalas 0.0.1
abstract pandas utilities
"""
import os
import sys
import pandas
import numpy as np
import bokeh
from bokeh import charts, io
modself = sys.modules[__name__]

class AbstractFrame(object):
  """
  Abstract frame wrapper
  Passes ALL to self.frame
  Can be initialized with pandas.DataFrame, pandas.Series, koalas.AbstractFrame or just lists of lists
  Adds columns as attributes so that frame.column works as well as frame['column'] (get,set&del)
  Does some series magic so when doing frame[pandas.Series] it casts as self.__class__
  """
  def __init__(self, frame=None, **options):
    if type(frame) in (pandas.DataFrame, pandas.core.series.Series) or frame: # check that frame is not none (without trying to bool(DataFrame/Series))
      if frame.__class__ == pandas.core.series.Series: frame = pandas.DataFrame(frame)
      elif frame.__class__ == pandas.DataFrame: pass
      elif isinstance(frame, AbstractFrame):
        opts = frame._options
        opts.update(options)
        options = opts
        frame = frame.frame
      else: frame = pandas.DataFrame(frame)
    else: frame = pandas.DataFrame()
    self.frame = frame
    self._options = options
    self._callback('init')

  def __getitem__(self, key):
    if key.__class__ == pandas.core.series.Series: return self.cast(self.frame[key])
    return self.frame[key]

  def __setitem__(self, key, val):
    self.frame[key] = val

  def __delitem__(self, key):
    del self.frame[key]

  def __getattr__(self, key):
    if key in self.frame.columns: return self.frame[key]
    if key == 'peek': return self.frame.iloc[0:5]
    return self.frame.__getattribute__(key)

  def __setattr__(self, key, val):
    if key == 'frame' or key[0] == '_': object.__setattr__(self, key, val)
    elif key in self.frame.columns: self.frame[key] = val
    else:
      if key in self.frame.__dict__: self.frame.__setattr__(self, key, val) # @TODO still doesnt work for af.columns=
      else: self.frame.__setitem__(key, val)

  def __delattr__(self, key):
    self.frame.__delitem__(key)

  def __dir__(self):
    return self.__dict__.keys()+list(self.frame.columns)+dir(self.frame)

  def _callback(self, cbname, **kwargs):
    meth = getattr(self, 'on%s%s'%(cbname[0].upper(), cbname[1:]), None)
    if callable(meth): meth(**kwargs)

  def load(self, **kwargs):
    self._callback('load', **kwargs)
    return self

  def cast(self, frame, **options):
    "Cast as same class as self, with options inherited"
    opts = {}
    opts.update(self._options)
    opts.update(options)
    return self.__class__(frame, **opts)

  def asaf(self, type=None):
    if not type: type = AbstractFrame
    if not isinstance(self, type):
      return type(self.frame)
    return self

  def capply(self, *args, **kwargs):
    return self.cast(self.frame.apply(*args, **kwargs))


# monkeypatch
def asaf(self, type=AbstractFrame):
  return type(self)
pandas.DataFrame.asaf = asaf

def firstDayOfWeek(date):
  "Assuming monday"
  return (date - pandas.Timedelta(days=date.weekday())).date()

class UtilMixin():
  # utilities
  def takeIndexes(self, index):
    "Filter frame by given index"
    return self.loc[index] # much faster (test it :D)
    #return self[self.frame.apply(lambda row: row.name in index, axis=1)]

  def groupSizeSort(self, field, ascending=False):
    "Group by given field, size() and sort (ascending=False by default)"
    grouped = self.frame.groupby(field).size()
    grouped.sort(ascending=ascending, inplace=True)
    return grouped

  def occurencePerColumn(self, val):
    return self.transpose().apply(lambda row: float(len(row[row==val]))/len(row[row.notnull()]), axis=1)

  def valuesPerColumn(self):
    return self.apply(lambda col: col.value_counts(), axis=0)

  def booleanize(self):
    "Replace all values greater than 0 with 1"
    def rowToBitArray(row):
      return map(lambda d: d if pandas.isnull(d) else (1 if d > 0 else 0), row)
    return self.cast(self.frame.apply(rowToBitArray, axis=1))


class NumberMixin():
  "Utils for frames that have series of numbers (e.g. counts or users' launches/week)"

  #def percentiles(self, field, percentiles=10):
  #  pass

  def bins(self, field, bins=30):
    "Apply bin categories to a numeric series, returns (label, appliedSeries)"
    fld = self.frame[field]
    mn, mx = fld.min(), fld.max()

    binsize = (float(mx-mn)/bins)
    binedges = map(lambda i: mn+(i*binsize), range(bins+1))
    labels = map(lambda x: x+(binsize/2), binedges[:-1])

    def getbin(val):
      i = int(np.ceil((val-mn)/binsize))-1
      return labels[i if i > -1 else 0]

    return labels, fld.apply(getbin)

  def binCounts(self, field, bins=30):
    labels, applied = self.bins(field, bins)
    applied = NumberFrame(applied)
    counts = pandas.DataFrame(labels)
    counts.set_index(0, inplace=True)
    counts['num'] = applied.groupby(0).size()
    counts['num'].fillna(0, inplace=True)
    return counts

class NumberFrame(AbstractFrame, NumberMixin, UtilMixin):
  pass

class EventsMixin():
  "Utils for frames that are series of events (e.g. logs)"
  def onLoad(self, datetime_field='datetime'):
    self._options = {'datetime_field': datetime_field}
    self.frame[datetime_field] = pandas.to_datetime(self.frame[datetime_field])
    #self.frame['date'] = self.frame[datetime_field].apply(lambda d: d.date())

  def periodCount(self, groupby, period='daily'):
    """
    Count events by groupby field in daily/weekly/monthly resolution
    Returns a frame with groupby field as indexes and a column for each period
    """
    frame = self.frame.copy()
    if period == 'hourly':
      def hour(d):
        return pandas.Timestamp(d.strftime('%Y-%m-%d %H'))
      frame['hour'] = frame[self._options['datetime_field']].apply(hour)
      counts = frame.groupby([groupby, 'hour']).size().unstack()
      counts.fillna(0, inplace=True)

      mn = frame['hour'].min()
      mx = frame['hour'].max()

      print mn, mx
      for d in pandas.date_range(start=mn, end=mx, freq='H'):
        d = d + pandas.Timedelta('1 hour')
        if not d in counts.columns:
          counts[d] = 0

      return TimeFrame(counts).load()

    elif period == 'daily':
      frame['date'] = frame[self._options['datetime_field']].apply(lambda d: d.date())
      counts = frame.groupby([groupby, 'date']).size().unstack()
      counts.fillna(0, inplace=True)

      # blow up (fill dates that have no events with zeroes)
      d = frame['date'].min()
      mx = frame['date'].max()
      while d < mx:
        d = d + pandas.Timedelta('1 day')
        if not d in counts.columns:
          counts[d] = 0

      return TimeFrame(counts).load()

    elif period == 'weekly':
      frame['week'] = frame[self._options['datetime_field']].apply(firstDayOfWeek)
      counts = frame.groupby([groupby, 'week']).size().unstack()
      counts.fillna(0, inplace=True)

      # blow up (fill weeks that have no events with zeroes)
      mn = frame['week'].min()
      mx = frame['week'].max()
      for d in pandas.date_range(start=mn, end=mx, freq='W-MON'):
        d = d.date()
        #d = d.strftime('%Y-%W')
        if not d in counts.columns:
          counts[d] = 0

      return TimeFrame(counts).load()

    else: raise Exception("EventsMixin.periodCount period not implemented: %s"%period)

class EventsFrame(AbstractFrame, EventsMixin, UtilMixin):
  pass

class TimeMixin():
  "Mixin for frames that have dates as columns (e.g. counts of events for each day for users)"

  def onLoad(self):
    # sort columns
    self.frame = self.frame.reindex_axis(sorted(self.frame.columns), axis=1)

  def min(self):
    return self.frame.columns.min()

  def max(self):
    return self.frame.columns.max()

  def maskInvalid(self, min_date=None, max_date=None):
    min_series = isinstance(min_date, pandas.Series)
    max_series = isinstance(max_date, pandas.Series)
    if not min_series and not min_date: min_date = self.min()
    if not max_series and not max_date: max_date = self.max()

    def doMask(row):
      return row[min_date[row.name] if min_series else min_date:max_date[row.name] if max_series else max_date]

    return self.cast(self.frame.apply(doMask, axis=1))

  def toOffsets(self, nan=np.nan):
    "Replace columns as days from beginning, beginning taken for each device the next day/week of last NaN"
    def deltize_row(row):
      notnull = row[row.notnull()] if pandas.isnull(nan) else row[row != nan]
      notnull.index = range(0,len(notnull))
      return notnull
    return self.capply(deltize_row, axis=1)

  def classify(self, date, *config):
    """
    Classify indexes by evaluating a (bool)function with sum of counts in a certain period before given date
    e.g. classify(
      ('NaN', '30 days', lambda events: np.isnull(events)),
      ('heavy', '3 days', lambda events: events > 20),
      ('regular', '7 days', lambda events: events > 10),
      ('dead', '30 days', lambda events: events == 0),
      ('idle',None,None)
    )
    """
    print '.',
    sys.stdout.flush()
    # sum events for every period given in config
    periods = NumberFrame(pandas.DataFrame(index=self.frame.index))
    for period in set(map(lambda line: line[1] if len(line) > 1 else None, config)):
      if not period: continue # Nones
      start = date - pandas.Timedelta(period) + pandas.Timedelta('1 day')
      periods[period] = self.frame.loc[self.frame.index, start:date].sum(axis=1, skipna=False)

    def cls(row):
      for cls, period, func in config:
        if not period: return cls
        if func(periods.loc[row.name, period]): return cls

    return periods.apply(cls, axis=1)

  @classmethod
  def newDaily(self, start_date, days, index=None):
    frame = TimeFrame(pandas.DataFrame(index=index))
    for date in pandas.date_range(start_date, periods=days):
      frame[date.date()] = np.nan
    return frame

  @classmethod
  def newWeekly(self, start_date, weeks, index=None):
    frame = TimeFrame(pandas.DataFrame(index=index))
    for date in pandas.date_range(start_date, periods=weeks):
      frame[date.strftime('%Y-%W')] = np.nan
    return frame

class TimeFrame(AbstractFrame, TimeMixin, NumberMixin, UtilMixin):
  pass

class BokehMixin():
  def _chart(self, chartcls, **kwargs):
    opts = dict(width=1000, height=500, legend='bottom_left')
    show = kwargs.pop('show', True)
    opts.update(self.kwargs())
    opts.update(kwargs)
    p = chartcls(self.frame, **opts)
    if show: charts.show(p)
    else: charts.save(p)
    return p

  def line(self, **kwargs):
    return self._chart(charts.Line, **kwargs)

  def bar(self, **kwargs):
    return self._chart(charts.Bar, **kwargs)

  def time(self, **kwargs):
    return self._chart(charts.TimeSeries, **kwargs)

  def area(self, **kwargs):
    return self._chart(charts.Area, **kwargs)

  def kwargs(self):
    return {}

class BokehFrame(AbstractFrame, BokehMixin, NumberMixin, UtilMixin):
  pass

class BokehGroups():
  def __init__(self, keyFunc, basename, **kwargs):
    self.files = dict()
    self.basename = basename
    self.keyFunc = keyFunc
    self.kw = dict()
    self.kw.update(dict(mode='cdn'))
    self.kw.update(kwargs)

  def setOutput(self, comb):
    gr = self.keyFunc(comb)
    if gr not in self.files: self.files[gr] = []
    fn = '%s_%s_%s.html'%(self.basename, gr, len(self.files[gr]))
    self.files[gr].append(fn)
    io.output_file(fn, **self.kw)
    return fn

  def glue(self, clean=True):
    for group, files in self.files.items():
      fn = '%s_%s.html'%(self.basename, group)
      f = open(fn, 'w')
      for src in files:
        s = open(src, 'r')
        f.write(s.read())
        s.close()
        os.unlink(src)
      f.close()

class ArgCombGraph(BokehFrame):
  def onLoad(self, name=None, comb=None):
    self._options['name'] = name
    self._options['comb'] = comb

  def kwargs(self):
    return dict(title=self._options['name'])

# utilities @TODO tests, move to another file, something..

class cacher(object):
  "Cache decorator"
  def __init__(self, name, load_columns=None, asaf=AbstractFrame, method=True, rootdir='cache/'):
    self.name = name
    self.load_columns = load_columns
    self.asaf = asaf
    self.method = method
    self.rootdir = rootdir

  def _filename(self, args, kwargs):
    if self.method: args = args[1:]
    parts = map(self._cleanArg, [self.name]+list(args)+map(lambda k: kwargs[k], sorted(kwargs.keys())))
    return self.rootdir + '_'.join(parts) + '.csv'

  def _cleanArg(self, arg):
    if type(arg) in (list, tuple): return '-'.join(map(str, arg))
    return str(arg)

  def __call__(self, func):
    def wrapped(*args, **kwargs):
      filename = self._filename(args, kwargs)
      if os.path.exists(filename):
        print 'Loading from %s'%filename
        frame = pandas.io.parsers.read_csv(filename, index_col=0).asaf(self.asaf)
        if self.load_columns:
          frame.frame.columns = frame.frame.apply(lambda col: self.load_columns(col.name), axis=0)
        return frame
      else:
        print 'Generating %s'%filename
        frame = func(*args, **kwargs)
        frame.to_csv(filename)
        return frame
    return wrapped

def pairs(k, vs): return map(lambda v: (k,v), vs)

import itertools
class ArgComb():
  def __init__(self, name, **kwargs):
    self.name = name
    self.kw = dict()
    self.kwd = dict()
    self.kwpersist = kwargs
    self.dimensions = []
    self.conditionals = []

  def conditional(self, func):
    self.conditionals.append(func)

  def add(self, name, *values, **kwargs):
    self._frames = []
    self.dimensions.append(name)
    self.kw[name] = values
    display = kwargs.pop('display', None)
    if type(display) == list:
      if len(display) != len(values): raise Exception('Display length mismatch')
      self.kwd[name] = display

  def _name(self, comb):
    def dimension_display(d):
      dd = d.split('_')
      dd = ''.join(map(lambda s: s[0], dd[:-1]))+dd[-1]
      dd = dd[0:15]
      if comb[d] == None: return None
      vd = str(comb[d])[0:10]
      if d in self.kwd:
        vd = self.kwd[d][self.kw[d].index(comb[d])]
        self.kw[d].index(comb[d])
      return "%s=%s"%(dd, vd)
    return self.name + '; ' + ', '.join(filter(bool, map(dimension_display, self.dimensions)))

  def __iter__(self):
    def append(dct):
      d = dict()
      d.update(self.kwpersist)
      d.update(dict(dct))
      for cond in self.conditionals:
        d.update(cond(d))
      return d
    for c in map(append, itertools.product(*map(lambda d: pairs(d, self.kw[d]), self.dimensions))):
      yield self._name(c), c

  def apply(self, func):
    l = len(list(self))
    i = 0
    for name, c in self:
      i += 1
      print "### (%s/%s) %s"%(i, l, name)
      yield name, c, func(**c)
      print

  def charts(self, frames, group_func, basename, cls, chartname='line', **kwargs):
    fr = []
    bg = BokehGroups(group_func, basename)
    for name, c, frame in (self.apply(frames) if callable(frames) else frames):
      fr.append((name, c, frame))
      bg.setOutput(c)
      graph = cls(frame).load(name=name, comb=c)
      getattr(graph, chartname)(**kwargs)
    bg.glue()
    return fr

  def __str__(self):
    return "ArgComb(%s): %s"%(len(list(self)), ', '.join(map(lambda d: "%s(%s)"%(d, len(self.kw[d])), self.dimensions)))

  def __repr__(self):
    return str(self)

