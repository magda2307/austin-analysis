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
    
    time.sleep(10)  # wait for streamlit to start
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1600, "height": 1000})
            page = context.new_page()
            
            print("Navigating to http://localhost:8501...")
            page.goto("http://localhost:8501", timeout=60000)
            
            print("Waiting for .stApp...")
            page.wait_for_selector(".stApp", state="visible", timeout=60000)
            print("Waiting 15 seconds for models to load...")
            time.sleep(15)
            
            os.makedirs("reports/figures", exist_ok=True)
            
            print("Selecting 'Polski' language...")
            try:
                # Click the language selectbox to open the dropdown
                page.locator('div[data-baseweb="select"]').first.click(timeout=5000)
                time.sleep(1)
                # Click the 'Polski' option
                page.locator('li[role="option"]', has_text="Polski").click(timeout=5000)
                print("Switched language to Polski. Waiting for re-render...")
                time.sleep(5)
            except Exception as e:
                print(f"Could not change language: {e}")
            
            
            tabs = page.locator('button[data-baseweb="tab"]')
            count = tabs.count()
            print(f"Found {count} tabs.")
            
            for i in range(count):
                tabs.nth(i).click()
                time.sleep(3)
                page.screenshot(path=f"reports/figures/streamlit_ui_tab{i}_full.png", full_page=True)
                print(f"Saved reports/figures/streamlit_ui_tab{i}_full.png")
            
            browser.close()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Killing Streamlit...")
        process.terminate()
        process.wait()

if __name__ == "__main__":
    main()
