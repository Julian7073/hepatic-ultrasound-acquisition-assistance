@echo off
pushd "%~dp0.."
python -m streamlit run ".\Codigos_Pipeline_Experimental_Segmentacion\gui_longitudinal.py" --server.port 8501 --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false
popd
