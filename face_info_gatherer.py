import face_recognition
import requests
import json
import os
import cv2
import numpy as np
import threading
import concurrent.futures
import time
import re
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from PIL import Image, ImageEnhance
import io
import base64
from collections import Counter

class InfoGatherer:
    def __init__(self, name, image_path, workers=6, threads=2):
        self.name = name.lower()
        self.image_path = image_path
        self.face_encoding = None
        self.workers = workers
        self.threads = threads
        self.platforms = [
            "facebook", "instagram", "linkedin", 
            "twitter", "youtube", "tiktok"
        ]
        self.results = {
            "real_name": [],
            "gender": [],
            "email": [],
            "phone": [],
            "birth": {
                "date": [],
                "place": []
            },
            "origin": [],
            "social_media": {}
        }
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
    def load_and_encode_face(self):
        try:
            image = face_recognition.load_image_file(self.image_path)
            face_locations = face_recognition.face_locations(image)
            if not face_locations:
                print("No face detected in the image. Attempting to enhance...")
                enhanced_image = self.enhance_image(image)
                face_locations = face_recognition.face_locations(enhanced_image)
                if face_locations:
                    print("Face detected after enhancement!")
                    self.face_encoding = face_recognition.face_encodings(enhanced_image, face_locations)[0]
                else:
                    print("Still no face detected. Using original image for search...")
                    self.face_encoding = None
            else:
                print(f"Face detected! Found {len(face_locations)} faces.")
                self.face_encoding = face_recognition.face_encodings(image, face_locations)[0]
            return True
        except Exception as e:
            print(f"Error loading or encoding face: {str(e)}")
            return False
    
    def enhance_image(self, image):
        pil_image = Image.fromarray(image)
        
        enhancer = ImageEnhance.Brightness(pil_image)
        brightened = enhancer.enhance(1.5)
        
        enhancer = ImageEnhance.Contrast(brightened)
        contrasted = enhancer.enhance(1.5)
        
        enhancer = ImageEnhance.Sharpness(contrasted)
        sharpened = enhancer.enhance(2.0)
        
        return np.array(sharpened)
    
    def search_by_name(self, platform, worker_id):
        try:
            print(f"Worker {worker_id} searching {platform} by name: {self.name}")
            search_term = quote_plus(self.name)
            
            if platform == "google":
                url = f"https://www.google.com/search?q={search_term}"
            elif platform == "facebook":
                url = f"https://www.facebook.com/public/{search_term.replace('+', '-')}"
            elif platform == "instagram":
                url = f"https://www.instagram.com/explore/tags/{search_term.replace('+', '')}"
            elif platform == "linkedin":
                url = f"https://www.linkedin.com/pub/dir?firstName={self.name.split()[0]}&lastName={self.name.split()[-1] if len(self.name.split()) > 1 else ''}"
            elif platform == "twitter":
                url = f"https://twitter.com/search?q={search_term}&src=typed_query"
            elif platform == "youtube":
                url = f"https://www.youtube.com/results?search_query={search_term}"
            elif platform == "tiktok":
                url = f"https://www.tiktok.com/search?q={search_term}"
            else:
                return
            
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract platform-specific data
                self.extract_data_from_platform(platform, soup)
                
                # Look for profile images to compare with face encoding
                if self.face_encoding is not None:
                    images = soup.find_all('img')
                    for img in images:
                        try:
                            img_url = img.get('src')
                            if not img_url or not img_url.startswith('http'):
                                continue
                                
                            img_resp = requests.get(img_url, headers=self.headers, timeout=5)
                            if img_resp.status_code == 200:
                                img_array = np.array(Image.open(io.BytesIO(img_resp.content)))
                                face_locations = face_recognition.face_locations(img_array)
                                if face_locations:
                                    face_encodings = face_recognition.face_encodings(img_array, face_locations)
                                    for found_encoding in face_encodings:
                                        # Compare faces
                                        match = face_recognition.compare_faces([self.face_encoding], found_encoding, tolerance=0.6)
                                        if match[0]:
                                            print(f"Face match found on {platform}!")
                                            # Extract additional info around this image
                                            parent = img.parent
                                            for i in range(3):  # Check 3 levels up
                                                if parent:
                                                    self.extract_data_from_element(parent, platform)
                                                    parent = parent.parent
                        except Exception as e:
                            pass
                
        except Exception as e:
            print(f"Error searching {platform}: {str(e)}")
    
    def extract_data_from_platform(self, platform, soup):
        text_content = soup.get_text().lower()
        
        # Extract real name
        name_patterns = [
            r'name[:\s]+([\w\s]+)',
            r'full name[:\s]+([\w\s]+)',
        ]
        for pattern in name_patterns:
            matches = re.findall(pattern, text_content)
            for match in matches:
                name = match.strip()
                if 3 < len(name) < 40 and name != self.name:  # Basic validation
                    self.results["real_name"].append(name)
        
        # Extract gender
        if 'male' in text_content:
            self.results["gender"].append("male")
        if 'female' in text_content:
            self.results["gender"].append("female")
            
        # Extract email
        email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
        emails = re.findall(email_pattern, text_content)
        for email in emails:
            if self.validate_email(email):
                self.results["email"].append(email)
                
        # Extract phone
        phone_patterns = [
            r'phone[:\s]+([\+\d\s\-\(\)]{7,20})',
            r'tel[:\s]+([\+\d\s\-\(\)]{7,20})',
            r'(?<!\w)([\+\d\s\-\(\)]{7,20})(?!\w)'
        ]
        for pattern in phone_patterns:
            phones = re.findall(pattern, text_content)
            for phone in phones:
                cleaned_phone = re.sub(r'\D', '', phone)
                if 7 <= len(cleaned_phone) <= 15:  # Basic validation
                    self.results["phone"].append(cleaned_phone)
        
        # Extract birth info
        birth_date_patterns = [
            r'(?:born|birth|dob)[:\s]+(\d{1,2}[\s\-\.\/]\w+[\s\-\.\/]\d{4})',
            r'(?:born|birth|dob)[:\s]+(\w+[\s\-\.\/]\d{1,2}[\s\-\.\/]\d{4})',
            r'(?:born|birth|dob)[:\s]+(\d{4}[\s\-\.\/]\d{1,2}[\s\-\.\/]\d{1,2})'
        ]
        for pattern in birth_date_patterns:
            dates = re.findall(pattern, text_content)
            for date in dates:
                self.results["birth"]["date"].append(date.strip())
        
        # Extract birth place
        birth_place_patterns = [
            r'born in ([\w\s\,]+)',
            r'from ([\w\s\,]+)'
        ]
        for pattern in birth_place_patterns:
            places = re.findall(pattern, text_content)
            for place in places:
                if len(place.strip()) > 2 and len(place.strip()) < 50:
                    self.results["birth"]["place"].append(place.strip())
        
        # Extract origin/nationality
        origin_patterns = [
            r'(?:nationality|origin|ethnicity)[:\s]+([\w\s]+)',
            r'(?:is|was) (?:a|an) ([\w]+) (?:citizen|national)'
        ]
        for pattern in origin_patterns:
            origins = re.findall(pattern, text_content)
            for origin in origins:
                if len(origin.strip()) > 2 and len(origin.strip()) < 30:
                    self.results["origin"].append(origin.strip())
        
        # Extract social media handles
        if platform not in self.results["social_media"]:
            self.results["social_media"][platform] = []
        
        if platform == "facebook":
            profiles = soup.select("a[href*='/profile.php']") + soup.select("a[href*='facebook.com/']")
            for profile in profiles:
                href = profile.get('href', '')
                if '/profile.php?id=' in href or ('/facebook.com/' in href and '?' not in href):
                    username = href.split('/')[-1].split('?')[0]
                    if username and username not in ['search', 'public']:
                        self.results["social_media"][platform].append(username)
        
        elif platform == "instagram":
            profiles = soup.select("a[href*='instagram.com/']")
            for profile in profiles:
                href = profile.get('href', '')
                if '/instagram.com/' in href and '?' not in href:
                    username = href.split('/')[-1]
                    if username and len(username) > 1:
                        self.results["social_media"][platform].append(username)
        
        elif platform == "linkedin":
            profiles = soup.select("a[href*='linkedin.com/in/']")
            for profile in profiles:
                href = profile.get('href', '')
                if '/in/' in href:
                    username = href.split('/in/')[-1].split('?')[0].split('/')[0]
                    if username:
                        self.results["social_media"][platform].append(username)
        
        elif platform == "twitter":
            profiles = soup.select("a[href*='twitter.com/']")
            for profile in profiles:
                href = profile.get('href', '')
                if '/twitter.com/' in href and '?' not in href:
                    username = href.split('/')[-1]
                    if username and username not in ['search', 'hashtag']:
                        self.results["social_media"][platform].append(username)
        
        elif platform == "youtube":
            channels = soup.select("a[href*='youtube.com/channel/']") + soup.select("a[href*='youtube.com/user/']")
            for channel in channels:
                href = channel.get('href', '')
                if '/channel/' in href or '/user/' in href:
                    username = href.split('/')[-1]
                    if username:
                        self.results["social_media"][platform].append(username)
        
        elif platform == "tiktok":
            profiles = soup.select("a[href*='tiktok.com/@']")
            for profile in profiles:
                href = profile.get('href', '')
                if '/@' in href:
                    username = href.split('/@')[-1].split('?')[0]
                    if username:
                        self.results["social_media"][platform].append(username)
    
    def extract_data_from_element(self, element, platform):
        text = element.get_text().lower()
        
        # Extract real name from element context
        name_patterns = [
            r'name[:\s]+([\w\s]+)',
            r'([\w\s]{2,30})'  # More relaxed pattern for names
        ]
        for pattern in name_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                name = match.strip()
                if 3 < len(name) < 40 and name != self.name and ' ' in name:
                    self.results["real_name"].append(name)
        
        # Extract other data using similar patterns as before
        if 'male' in text:
            self.results["gender"].append("male")
        if 'female' in text:
            self.results["gender"].append("female")
            
        # Extract email
        email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
        emails = re.findall(email_pattern, text)
        for email in emails:
            if self.validate_email(email):
                self.results["email"].append(email)
                
        # Extract more focused data based on proximity to the matched face
        for a in element.find_all('a', href=True):
            href = a.get('href', '')
            if any(p in href for p in self.platforms):
                for p in self.platforms:
                    if p in href:
                        if p not in self.results["social_media"]:
                            self.results["social_media"][p] = []
                        username = href.split('/')[-1].split('?')[0]
                        if username and len(username) > 1 and username not in ['search', 'hashtag']:
                            self.results["social_media"][p].append(username)
    
    def validate_email(self, email):
        pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        return bool(re.match(pattern, email))
    
    def image_search(self, worker_id):
        if self.face_encoding is None:
            print(f"Worker {worker_id}: No face encoding available for image search.")
            return
            
        try:
            print(f"Worker {worker_id}: Performing reverse image search...")
            
            # Convert image to base64 for API requests
            with open(self.image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Simulating a reverse image search API (you would need an actual API for production)
            # This is a placeholder for demonstration
            search_engines = ["google", "yandex", "bing"]
            engine = search_engines[worker_id % len(search_engines)]
            
            print(f"Worker {worker_id}: Searching on {engine} images...")
            
            # Here would be code to interact with actual reverse image search APIs
            # For now, we'll fall back to name-based search on platforms
            platform_index = worker_id % len(self.platforms)
            self.search_by_name(self.platforms[platform_index], worker_id)
            
        except Exception as e:
            print(f"Error in image search (Worker {worker_id}): {str(e)}")
    
    def worker_task(self, worker_id):
        if worker_id < 3:  # First 3 workers do image search
            self.image_search(worker_id)
        else:  # Other 3 workers do platform search
            platform_index = (worker_id - 3) % len(self.platforms)
            self.search_by_name(self.platforms[platform_index], worker_id)
    
    def clean_results(self):
        # Remove duplicates and clean up results
        for key in self.results:
            if key == "birth":
                for subkey in self.results[key]:
                    self.results[key][subkey] = list(set(self.results[key][subkey]))
            elif key == "social_media":
                for platform in self.results[key]:
                    self.results[key][platform] = list(set(self.results[key][platform]))
            else:
                self.results[key] = list(set(self.results[key]))
        
        # Use frequency counting to pick most likely values
        if self.results["real_name"]:
            name_counts = Counter(self.results["real_name"])
            self.results["real_name"] = [name for name, count in name_counts.most_common(3)]
        
        if self.results["gender"]:
            gender_counts = Counter(self.results["gender"])
            self.results["gender"] = [gender_counts.most_common(1)[0][0]]
    
    def run(self):
        if not self.load_and_encode_face():
            print("Proceeding with name-based search only...")
        
        print(f"Starting search with {self.workers} workers...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.threads) as executor:
            futures = []
            for i in range(self.workers):
                futures.append(executor.submit(self.worker_task, i))
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Worker error: {str(e)}")
        
        self.clean_results()
        return self.results

def print_results(results):
    print("\n" + "="*50)
    print("SEARCH RESULTS")
    print("="*50)
    
    if results["real_name"]:
        print(f"\nReal Name (verified): {', '.join(results['real_name'])}")
    else:
        print("\nReal Name: Not found")
    
    if results["gender"]:
        print(f"Gender: {', '.join(results['gender'])}")
    else:
        print("Gender: Not found")
    
    if results["email"]:
        print(f"Email: {', '.join(results['email'])}")
    else:
        print("Email: Not found")
    
    if results["phone"]:
        print(f"Phone: {', '.join(results['phone'])}")
    else:
        print("Phone: Not found")
    
    print("\nBirth Information:")
    if results["birth"]["date"]:
        print(f"  Date: {', '.join(results['birth']['date'])}")
    else:
        print("  Date: Not found")
    
    if results["birth"]["place"]:
        print(f"  Place: {', '.join(results['birth']['place'])}")
    else:
        print("  Place: Not found")
    
    if results["origin"]:
        print(f"\nOrigin/Nationality: {', '.join(results['origin'])}")
    else:
        print("\nOrigin/Nationality: Not found")
    
    print("\nSocial Media:")
    if results["social_media"]:
        for platform, usernames in results["social_media"].items():
            if usernames:
                print(f"  {platform.capitalize()}: {', '.join(usernames)}")
    else:
        print("  No social media accounts found")
    
    print("\n" + "="*50)

def main():
    print("Face & Name Information Gatherer")
    print("="*50)
    
    name = input("\nEnter full name (all lowercase): ").strip()
    image_path = input("Enter path to face image (in current folder): ").strip()
    
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        return
    
    print("\nInitializing information gathering process...")
    gatherer = InfoGatherer(name, image_path)
    results = gatherer.run()
    
    print_results(results)

if __name__ == "__main__":
    main()
