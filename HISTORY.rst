=======
History
=======

0.1.16 (2018-03.15)
------------------

* I52 symmetry bug fix

0.1.15 (2018-03.05)
------------------

* add NullCriteria that always returns 0 err


0.1.14 (2018-02-28)
------------------

* fix provenance bug in 'cyclic entry' cases
* try to make serialization of Segments more efficient

0.1.13 (2018-02-16)
------------------

* raise exception if system too large

0.1.12 (2018-02-16)
------------------

* partial bignum fix

0.1.11 (2018-02-15)
------------------

* fix memory bug
* make distribution work better
* maybe fix pose bug, still some logic err, but maybe ok

0.1.10 (2018-02-15)
------------------

* add max_results option to grow
* fix C2 sym bug
* fix xform axis cen bug
* fix memory "bug" with batch parallel processing

0.1.9 (2018-02-08)
------------------

* add max_samples option to grow

0.1.8 (2018-02-07)
------------------

* origin_seg bug fix

0.1.6 (2018-02-01)
------------------

* middle-to-end cyclic fusions working
* add pretty logo of mid-to-end C3 fusion

0.1.6 (2018-02-01)
------------------

* bug fix in fullatom option

0.1.5 (2018-02-01)
------------------

* add fullatom option to Worms.sympose
* cyclic premutation working for simple beginning-to-end case

0.1.4 (2018-02-01)
------------------

* pypi deplolment derp

0.1.3 (2018-02-01)
------------------

* pypi deplolment derp

0.1.2 (2018-01-23)
------------------

* Add __main__ for module to run tests
* move worms.pdb to worms.data because pdb is kinda reserved
* move utility stuff to util.py
* add some interactive visualization utils for debugging

0.1.1 (2018-01-23)
------------------

* First release on PyPI.
