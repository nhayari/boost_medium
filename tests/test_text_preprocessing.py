import pandas as pd
import pytest
from utils.text_preprocessing import (
    strip_html_content,
    remove_non_ascii,
    to_lowercase,
    remove_numbers,
    remove_punctuation,
    remove_extra_whitespace,
    remove_stopwords,
    tokenize_text,
    stem_text,
    basic_text_clean,
    advanced_text_clean,
    preprocess_dataframe,
    create_combined_text_features,
)


@pytest.mark.skip(reason="ignoré car il ya un test general : advanced")
def test_strip_html_content():
    assert strip_html_content("<p>name <b>email</b></p>") == " name  email  "
    assert strip_html_content(None) == ""


@pytest.mark.skip(reason="ignoré car il ya un test general : advanced")
def test_to_lowercase():
    assert to_lowercase("HELLO") == "hello"
    assert to_lowercase(None) == ""

@pytest.mark.skip(reason="ignoré car il ya un test general : advanced")
def test_remove_numbers():
    assert remove_numbers("abc987") == "abc"
    assert remove_numbers("29 articles in the folder") == " articles in the folder"

@pytest.mark.skip(reason="ignoré car il ya un test general : advanced")
def test_remove_punctuation():
    assert remove_punctuation("medium, Hello!") == "medium Hello"
    assert remove_punctuation("that-is!punctuation.") == "thatispunctuation"


@pytest.mark.skip(reason="ignoré car il ya un test general : advanced")
def test_remove_extra_whitespace():
    assert remove_extra_whitespace("  no white space  ") == "no white space"
    assert remove_extra_whitespace("") == ""


@pytest.mark.skip(reason="ignoré car il ya un test general : advanced")
def test_remove_stopwords():
    text = "this is a medium"
    cleaned = remove_stopwords(text)
    assert "this" not in cleaned
    assert "medium" in cleaned

@pytest.mark.skip(reason="ignoré car il ya un test general : advanced")
def test_tokenize_text():
    assert tokenize_text("Medium Project") == ["Medium", "Project"]
    assert tokenize_text("") == []


@pytest.mark.skip(reason="ignoré car il ya un test general : advanced")
def test_stem_text():
    assert stem_text("running jumped") in ["run jump", "run jumped"]  # stemmer result may vary
    assert stem_text(None) == ""


# @pytest.mark.skip(reason=" active si besoin pour ignoer le test")
def test_basic_text_clean():
    text = "Running quickly!!! 123"
    result = basic_text_clean(text)
    assert "running" in result
    assert "123" not in result
    assert "!" not in result



# @pytest.mark.skip(reason="active si besoin pour ignoer le test")
def test_advanced_text_clean():
    html_text = "<p>Running medium project and compiling ! 123</p>"
    result = advanced_text_clean(html_text)
    assert "run" in result or "running" in result
    assert "compil" in result or "compiling" in result
    assert "123" not in result
    assert "<p>" not in result


@pytest.mark.skip(reason="ignoré temporairement")
def test_preprocess_dataframe():
    df = pd.DataFrame({"text": ["medium prediction!", "Running fast 123"]})
    df_processed = preprocess_dataframe(df, ["text"])
    assert "text_cleaned" in df_processed.columns
    assert all(isinstance(val, str) for val in df_processed["text_cleaned"])

@pytest.mark.skip(reason="ignoré temporairement")
def test_create_combined_text_features():
    df = pd.DataFrame({
        "title": ["Hello", "prediction expected"],
        "content": ["Medium", "is arriving.."]
    })
    df_processed = preprocess_dataframe(df, ["title", "content"])
    df_combined = create_combined_text_features(df_processed, ["title", "content"])
    assert "combined_col" in df_combined.columns
    assert isinstance(df_combined["combined_col"].iloc[0], str)
