STIXCore Products
*****************

The ``products`` submodule contains processing classes representing high level
data products created from multiple packets with additional checks.

The products are organized by processing level (``LB`` raw binary, ``L0``-``L3``)
and by product category (``ANC`` ancillary, ``CAL`` calibration):

.. toctree::
   :maxdepth: 2

   products/lb
   products/l0
   products/l1
   products/l2
   products/l3
   products/ll
   products/anc
   products/cal


Base classes
============

Shared base classes and helpers used by all product levels.

.. automodapi:: stixcore.products.product
    :include-all-objects:

.. automodapi:: stixcore.products.common
