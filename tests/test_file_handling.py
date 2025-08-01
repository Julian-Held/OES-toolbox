import pytest
from hypothesis import given, assume, example,strategies as st
from hypothesis.extra.numpy import arrays
from pathlib import Path
from OES_toolbox.file_handling import FileLoader
import pandas as pd
import numpy as np
import io

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xarray import DataArray,Dataset

from pandas.testing import assert_frame_equal, assert_index_equal
from numpy.testing import assert_allclose
from xarray.testing import assert_identical

tmp_file_names = ["comma_dot","tab_dot","tab_comma","semicolon_dot","semicolon_comma","bar_dot","bar_comma"]
encodings = ["utf-8","ascii","cp1252", "utf-16","utf-16be","utf-16le","utf-32","macroman"]

@st.composite
def separator_decimal_pair(draw):
    sep = draw(st.sampled_from(["\t", "|", " ", ";", ","]))
    decimal = draw(st.sampled_from([".", ","]))
    assume(sep != decimal)
    return sep, decimal

class TestFileLoader:

    @pytest.mark.parametrize(
        "line,valid,invalid",
        (
            ("1.0\t0.11", ("\t", "."), ("\t", ",")),
            ("1,0\t0,11", ("\t", ","), ("\t", ".")),
            ("1.0;2.0", (";", "."), (";", ",")),
            ("1,0;2,0", (";", ","), (";", ".")),
            ("1e2|2", ("|", "."), ("|", ",")),
            ("1,0e2|2,0", ("|", ","), ("|", ".")),
        ),
    )
    def test_infer_text_schema_from_line(self, line, valid, invalid):
        assert FileLoader._infer_text_schema_from_line(line) == valid
        assert FileLoader._infer_text_schema_from_line(line) != invalid


    @pytest.mark.parametrize("name, encoding", [(name, encoding) for name in tmp_file_names for encoding in encodings])
    def test_read_generic_text(self,temp_text_files, example_dataframe,name:str, encoding:str):
        f = temp_text_files.joinpath(f"{name}_{encoding}.txt")
        data_read = FileLoader._read_generic_text(f)
        assert_frame_equal(data_read,example_dataframe)

    
class TestFileLoader_PropertyBased:

    @given(separator_decimal_pair(),st.lists(st.floats(min_value=-1e3, max_value=1e12), min_size=2, max_size=12))
    def test_infer_text_schema_from_line_generative(self,pair,values):
        sep, decimal = pair
        line =  sep.join([f"{x:.3f}".replace(".",decimal) for x in values])
        infered_sep,infered_dec = FileLoader._infer_text_schema_from_line(line)
        assert sep == infered_sep 
        assert decimal == infered_dec
        # scientific notation
        line =  sep.join([f"{x:.3e}".replace(".",decimal) for x in values])
        infered_sep,infered_dec = FileLoader._infer_text_schema_from_line(line)
        assert sep == infered_sep 
        assert decimal == infered_dec

    @given(
        separator_decimal_pair(), 
        arrays(
            float,
            st.tuples(
                st.integers(min_value=2,max_value=10),
                st.integers(min_value=2,max_value=10)
            ),
            elements=st.floats(allow_nan=False,allow_infinity=False)
        )
    )
    def test_parse_open_text_file(self,pair,values):
        sep,decimal = pair
        buff = io.StringIO()
        # Make sure to exclude header, `_parse_open_text_file` starts reading in the data block
        pd.DataFrame(values).to_csv(buff,sep=sep, decimal=decimal,index=False, header=False)
        buff.seek(0)
        data = FileLoader._parse_open_text_file(buff,0,*pair)
        assert_allclose(data,values)
