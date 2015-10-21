import unittest
import pandas
import sys
import numpy as np
from koalas import *

class TestFrame(AbstractFrame): pass
class UtilFrame(AbstractFrame, UtilMixin): pass

TestFrame = AbstractFrame()

class AbstractFrameTest(unittest.TestCase):

  def setUp(self):
    pass

  def testInitialization(self):
    # create empty frame
    af = AbstractFrame()
    self.assertIsInstance(af.frame, pandas.DataFrame)

    # create frame from arrays
    af = AbstractFrame([[0,'a'], [1, 'b'], [2, 'c']])
    self.assertIsInstance(af.frame, pandas.DataFrame)

    # create frame from series
    af = AbstractFrame(af.frame[0])
    self.assertIsInstance(af.frame, pandas.DataFrame)

    # create frame from pandas.DataFrame
    af = AbstractFrame(af.frame)
    self.assertIsInstance(af.frame, pandas.DataFrame)

    # create frame from AbstractFrame
    af = AbstractFrame(af)
    self.assertIsInstance(af.frame, pandas.DataFrame)

  def testGetSetitem(self):
    af = AbstractFrame([[0,'a'], [1, 'b'], [2, 'c']])
    af.frame.columns = ('number', 'string')

    # get item (columns from frame)
    self.assertEqual(af['number'][0], 0)
    self.assertEqual(af['string'][1], 'b')

    # set item
    af['number'] = af['string']
    self.assertEqual(list(af['number']), list(af['string']))

  def testGetattr(self):
    # get attribute should proxy self.frame methods
    af = AbstractFrame([[0,'a'], [1, 'b'], [2, 'c']])
    self.assertEqual(af.groupby, af.frame.groupby)

  def tearDown(self):
    pass

class UtilFrameTest(unittest.TestCase):

  def setUp(self):
    self.uf = UtilFrame([[0,'a'], [1, 'b'], [2, 'c']])

  def testTakeIndexes(self):
    index = pandas.Index([0,2])
    self.assertEqual(list(self.uf.takeIndexes(index)[0]), [0,2])

  def testGroupSizeSort(self):
    self.uf = UtilFrame([[0,'a'], [1, 'b'], [2, 'c'], [3, 'a'], [4, 'b'], [5, 'a']])
    grouped = self.uf.groupSizeSort(1)
    self.assertEqual(list(grouped.index), ['a', 'b', 'c'])

  def tearDown(self):
    pass

class NumberFrameTest(unittest.TestCase):
  def setUp(self):
    self.nf = NumberFrame([[1, 'b'], [2, 'c'], [3, 'a'], [4, 'b']])

  def testBins(self):
    labels, applied = self.nf.bins(0, 2)    # 2 bins on column 0
    self.assertEqual(labels, [1.75, 3.25])  # values being 1,2,3,4 the two bins are [1, 2.5) [2.5, 4] (labels are means)
    self.assertEqual(list(applied), [1.75, 1.75, 3.25, 3.25])

  def testBinCounts(self):
    counts = self.nf.binCounts(0, 2)
    self.assertEqual(list(counts.index), [1.75, 3.25])
    self.assertEqual(list(counts['num']), [2, 2])

  def tearDown(self):
    pass

class EventFrameTest(unittest.TestCase):

  def setUp(self):
    self.ef = EventsFrame([
      ['a', '2015-09-25 00:00:00'],
      ['a', '2015-09-26 00:00:00'],
      ['a', '2015-09-26 00:00:00'],
      ['a', '2015-09-26 00:00:00'],
      ['b', '2015-09-27 00:00:00'],
      ['b', '2015-09-28 00:00:00'],
      ['b', '2015-09-28 00:00:00'],
      ['b', '2015-09-28 00:00:00'],
      ['a', '2015-09-29 00:00:00'],
      ['a', '2015-09-30 00:00:00'],
    ])
    self.ef.frame.columns = ['uuid', 'datetime']
    self.ef.load()

  def testPeriodCount(self):
    counts = self.ef.periodCount('uuid')
    self.assertEqual(list(counts[pandas.Timestamp('2015-09-25').date()]), [1,0])

  def tearDown(self):
    pass

class TimeFrameTest(unittest.TestCase):
  CLASSES = (
    (np.nan, '6 days', lambda e: pandas.isnull(e)),
    ('a', '3 days', lambda e: e >= 10),
    ('b', '6 days', lambda e: e > 10),
    ('c', None,     None)
  )

  def _days(self):
    return map(lambda d: pandas.Timestamp(d).date(),
      ['2015-09-25', '2015-09-26', '2015-09-27', '2015-09-28', '2015-09-29', '2015-09-30'])

  def setUp(self):
    self.tf = TimeFrame([
      [0, 10, 5, 0, 0, 5],
      [10, 5, 0, 0, 5, 0],
      [5, 0, 0, 5, 0, 10],
      [0, 0, 5, 10, 0, 0],
      [0, 5, 0, 0, 10, 0],
    ])
    self.tf.frame.columns = self._days()
    min_dates = self._days()[0:5]
    self.masked = self.tf.maskInvalid(min_date=pandas.Series(min_dates))

  def testMin(self):
    self.assertEqual(self.tf.min(), self._days()[0])

  def testMax(self):
    self.assertEqual(self.tf.max(), self._days()[-1])

  def testMaskInvalid(self):
    self.assertEqual(map(lambda v: pandas.isnull(v), self.masked.loc[3]), [True, True, True, False, False, False])

  def testBooleanize(self):
    booleaned = self.tf.booleanize()
    self.assertEqual(list(booleaned.frame.loc[0]), [0, 1, 1, 0, 0, 1])

  def testClassify(self):
    self.assertEqual(list(self.tf.classify(pandas.Timestamp('2015-09-30').date(), *self.CLASSES)), ['b', 'b', 'a', 'a', 'a'])
    self.assertEqual(list(self.masked.classify(pandas.Timestamp('2015-09-30').date(), *self.CLASSES)), ['b', np.nan, np.nan, np.nan, np.nan])

  def testToOffsets(self):
    # test with numbers
    #self.assertEqual(list(self.masked.toOffsets(nan=np.nan)[5]), [5.0, np.nan, np.nan, np.nan, np.nan])

    # test with classifications
    day = pandas.Timestamp('2015-09-30').date()
    classified = self.masked.classify(day, *self.CLASSES)
    tf = TimeFrame(pandas.DataFrame(classified, columns=[day]))
    self.assertEqual(list(tf.toOffsets()[0]), ['b', np.nan, np.nan, np.nan, np.nan])




#suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
#unittest.TextTestRunner(verbosity=2).run(suite)
