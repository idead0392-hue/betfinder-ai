# PrizePicks Props Integration - Quick Start

This is a quick reference guide for using the PrizePicks props integration in BetFinder AI.

## Quick Start (30 seconds)

### 1. Run the Demo
```bash
streamlit run demo_render_props.py
```

### 2. Basic Usage in Your Code
```python
from api_providers import PrizePicksProvider
from page_utils import render_props

# Fetch and render props
provider = PrizePicksProvider()
resp = provider.get_props(sport="basketball", max_props=25)

if resp.success:
    props = resp.data.get("data")
    render_props(props, top_n=25)
```

## Files Overview

| File | Purpose |
|------|---------|
| `api_providers.py` | Contains `PrizePicksProvider` class |
| `page_utils.py` | Contains `render_props()` function |
| `demo_render_props.py` | Interactive demo application |
| `example_prizepicks_usage.py` | Pattern from problem statement |
| `PRIZEPICKS_INTEGRATION.md` | Full documentation |
| `INTEGRATION_SUMMARY.md` | Implementation summary |

## Testing

```bash
# Run unit tests
python3 test_prizepicks_integration.py

# Run visual test
python3 test_visual_output.py

# Run Streamlit test app
streamlit run test_render_props.py
```

## Key Features

✅ Fetch props from PrizePicks data source  
✅ Display in clean Pandas DataFrame  
✅ Sport filtering  
✅ Summary statistics  
✅ Error handling with fallbacks  
✅ Formatted numeric displays  

## Common Use Cases

### Filter by Sport
```python
resp = provider.get_props(sport="basketball", max_props=50)
```

### Display Limited Props
```python
render_props(props_data, top_n=10)
```

### With Error Handling
```python
if resp.success:
    props = resp.data.get("data") if isinstance(resp.data, dict) else resp.data
    render_props(props, top_n=25)
else:
    st.error(f"Error: {resp.error_message}")
```

## Documentation

📖 **Full Guide**: See `PRIZEPICKS_INTEGRATION.md`  
📝 **Implementation Details**: See `INTEGRATION_SUMMARY.md`  
🎯 **Examples**: See `demo_render_props.py` and `example_prizepicks_usage.py`

## Support

For issues or questions:
1. Check the troubleshooting section in `PRIZEPICKS_INTEGRATION.md`
2. Review the example applications
3. Run the test suite to verify setup

## Next Steps

1. ✅ Install dependencies: `pip install streamlit pandas`
2. ✅ Run demo: `streamlit run demo_render_props.py`
3. ✅ Review documentation: `PRIZEPICKS_INTEGRATION.md`
4. ✅ Integrate into your pages using the examples
