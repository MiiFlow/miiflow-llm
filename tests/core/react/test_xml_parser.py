"""Tests for XML-based ReAct parser."""

import pytest
from miiflow_llm.core.react.parsing import XMLReActParser
from miiflow_llm.core.react.data import ReActParsingError


class TestXMLParser:
    """Test XML parser with various input formats."""

    def test_parse_tool_call_clean(self):
        """Test parsing valid tool call XML."""
        parser = XMLReActParser()
        response = """
        <thinking>I need to search for weather information in Paris</thinking>
        <tool_call>
            <name>weather</name>
            <input>{"city": "Paris", "country": "France"}</input>
        </tool_call>
        """

        result = parser.parse_response(response, 1)

        assert result.thought == "I need to search for weather information in Paris"
        assert result.action_type == "tool_call"
        assert result.action == "weather"
        assert result.action_input == {"city": "Paris", "country": "France"}
        assert not result.was_healed  # Clean parse, no healing needed

    def test_parse_final_answer_clean(self):
        """Test parsing final answer XML."""
        parser = XMLReActParser()
        response = """
        <thinking>I have enough information to provide a complete answer</thinking>
        <answer>The weather in Paris is currently 22°C and partly cloudy.</answer>
        """

        result = parser.parse_response(response, 1)

        assert result.thought == "I have enough information to provide a complete answer"
        assert result.action_type == "final_answer"
        assert result.answer == "The weather in Paris is currently 22°C and partly cloudy."
        assert not result.was_healed

    def test_parse_tool_call_with_markdown(self):
        """Test healing of XML wrapped in markdown code blocks."""
        parser = XMLReActParser()
        response = """
        ```xml
        <thinking>I need to calculate distance</thinking>
        <tool_call>
            <name>distance_calculator</name>
            <input>{"from": "NYC", "to": "LA"}</input>
        </tool_call>
        ```
        """

        result = parser.parse_response(response, 1)

        assert result.action_type == "tool_call"
        assert result.action == "distance_calculator"
        assert result.was_healed  # Regex extraction was used
        assert "regex_extraction" in result.healing_applied

    def test_parse_unclosed_thinking_tag(self):
        """Test healing of unclosed thinking tag."""
        parser = XMLReActParser()
        response = """
        <thinking>I need to search for information
        <tool_call>
            <name>search</name>
            <input>{"query": "test"}</input>
        </tool_call>
        """

        result = parser.parse_response(response, 1)

        assert result.action_type == "tool_call"
        assert result.was_healed
        assert "close_thinking_tag" in result.healing_applied or "regex_extraction" in result.healing_applied

    def test_parse_malformed_json_input(self):
        """Test handling of malformed JSON in input tag."""
        parser = XMLReActParser()
        response = """
        <thinking>Need to call tool</thinking>
        <tool_call>
            <name>search</name>
            <input>just a string, not JSON</input>
        </tool_call>
        """

        result = parser.parse_response(response, 1)

        assert result.action_type == "tool_call"
        assert result.action == "search"
        # Should fall back to wrapping in dict
        assert "input" in result.action_input
        assert result.action_input["input"] == "just a string, not JSON"

    def test_parse_missing_thinking_tag(self):
        """Test error when thinking tag is missing."""
        parser = XMLReActParser()
        response = """
        <tool_call>
            <name>search</name>
            <input>{"query": "test"}</input>
        </tool_call>
        """

        with pytest.raises(ReActParsingError, match="Missing <thinking>"):
            parser.parse_response(response, 1)

    def test_parse_missing_tool_name(self):
        """Test error when tool name is missing."""
        parser = XMLReActParser()
        response = """
        <thinking>Need to search</thinking>
        <tool_call>
            <input>{"query": "test"}</input>
        </tool_call>
        """

        with pytest.raises(ReActParsingError, match="Missing <name>"):
            parser.parse_response(response, 1)

    def test_parse_empty_answer(self):
        """Test error when answer is empty."""
        parser = XMLReActParser()
        response = """
        <thinking>Done thinking</thinking>
        <answer></answer>
        """

        with pytest.raises(ReActParsingError, match="Empty <answer>"):
            parser.parse_response(response, 1)

    def test_parse_mixed_content_extraction(self):
        """Test extraction from mixed natural language and XML."""
        parser = XMLReActParser()
        response = """
        Let me help you with that.

        <thinking>I need to search for this information</thinking>
        <tool_call>
            <name>search</name>
            <input>{"query": "test"}</input>
        </tool_call>

        I'll execute that now.
        """

        result = parser.parse_response(response, 1)

        assert result.action_type == "tool_call"
        assert result.action == "search"
        assert result.was_healed
        assert "extract_from_mixed" in result.healing_applied or "regex_extraction" in result.healing_applied

    def test_parse_case_insensitive(self):
        """Test that parser is case-insensitive for tags."""
        parser = XMLReActParser()
        response = """
        <THINKING>Need to search</THINKING>
        <TOOL_CALL>
            <NAME>search</NAME>
            <INPUT>{"query": "test"}</INPUT>
        </TOOL_CALL>
        """

        result = parser.parse_response(response, 1)

        assert result.action_type == "tool_call"
        assert result.action == "search"

    def test_parse_multiline_answer(self):
        """Test parsing answer with multiple lines."""
        parser = XMLReActParser()
        response = """
        <thinking>I have the information needed</thinking>
        <answer>
        Here is a comprehensive answer:
        • Point 1: First detail
        • Point 2: Second detail
        • Point 3: Third detail

        This covers everything.
        </answer>
        """

        result = parser.parse_response(response, 1)

        assert result.action_type == "final_answer"
        assert "Point 1" in result.answer
        assert "Point 2" in result.answer
        assert "Point 3" in result.answer

    def test_parse_complex_nested_json(self):
        """Test parsing tool input with nested JSON."""
        parser = XMLReActParser()
        response = """
        <thinking>Need to execute complex query</thinking>
        <tool_call>
            <name>database_query</name>
            <input>{
                "table": "users",
                "filters": {
                    "age": {"gt": 18, "lt": 65},
                    "active": true
                },
                "sort": ["name", "asc"]
            }</input>
        </tool_call>
        """

        result = parser.parse_response(response, 1)

        assert result.action_type == "tool_call"
        assert result.action == "database_query"
        assert result.action_input["table"] == "users"
        assert result.action_input["filters"]["age"]["gt"] == 18
        assert result.action_input["filters"]["active"] is True


class TestXMLParserEdgeCases:
    """Test edge cases and error handling."""

    def test_completely_invalid_xml(self):
        """Test that completely invalid input raises error."""
        parser = XMLReActParser()
        response = "This is just plain text with no XML tags at all"

        with pytest.raises(ReActParsingError):
            parser.parse_response(response, 1)

    def test_partial_xml_fragments(self):
        """Test handling of incomplete XML fragments."""
        parser = XMLReActParser()
        response = "<thinking>I need to"  # Incomplete

        with pytest.raises(ReActParsingError):
            parser.parse_response(response, 1)

    def test_get_parser_name(self):
        """Test parser name method."""
        parser = XMLReActParser()
        assert parser.get_name() == "XMLReActParser"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
