import subprocess
import time
import os
from playwright.sync_api import sync_playwright

def main():
    print("Starting Streamlit app...")
    process = subprocess.Popen(
        ["streamlit", "run", "streamlit_app.py", "--server.headless", "true"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    
    time.sleep(15)  # wait for streamlit to start and compile python
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1400, "height": 900})
            page = context.new_page()
            
            print("Navigating to http://localhost:8501...")
            page.goto("http://localhost:8501", timeout=60000)
            
            print("Waiting for .stApp...")
            page.wait_for_selector(".stApp", state="visible", timeout=60000)
            print("Waiting 15 seconds for models and graphs to load...")
            time.sleep(15)
            
            os.makedirs("reports/figures", exist_ok=True)
            
            # Screenshot 1: Default view
            page.screenshot(path="reports/figures/streamlit_ui_main.png")
            print("Saved reports/figures/streamlit_ui_main.png")
            
            # Try to click some tabs to get different views
            # Streamlit tabs are usually buttons inside a div with data-testid="stTabs"
            tabs = page.locator('button[data-baseweb="tab"]')
            count = tabs.count()
            print(f"Found {count} tabs.")
            
            if count > 1:
                tabs.nth(1).click()
                time.sleep(5)
                page.screenshot(path="reports/figures/streamlit_ui_tab2.png")
                print("Saved reports/figures/streamlit_ui_tab2.png")
                
            if count > 2:
                tabs.nth(2).click()
                time.sleep(5)
                page.screenshot(path="reports/figures/streamlit_ui_tab3.png")
                print("Saved reports/figures/streamlit_ui_tab3.png")
            
            browser.close()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Killing Streamlit...")
        process.terminate()
        process.wait()

if __name__ == "__main__":
    main()
