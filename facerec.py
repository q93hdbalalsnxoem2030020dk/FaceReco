import os
import requests
import bs4
import cv2
import PIL.Image
import concurrent.futures
import urllib.parse
import re
import io
import numpy as np
import face_recognition
import time

def analyze_face(image_path):
    try:
        img_pil = PIL.Image.open(image_path).convert('RGB')
        img_np = np.array(img_pil)

        face_locations = face_recognition.face_locations(img_np)
        if not face_locations:
            return False, "No face detected in the image by face_recognition"

        face_encoding = face_recognition.face_encodings(img_np, face_locations)[0]

        try:
            img_cv = cv2.imread(image_path)
            if img_cv is not None:
                 img_cv_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                 face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                 faces_cv = face_cascade.detectMultiScale(img_cv_gray, 1.1, 4)
                 face_detected_cv = len(faces_cv) > 0
            else:
                 face_detected_cv = False
        except Exception:
            face_detected_cv = False


        return True, {
            "face_detected_fr": True,
            "face_encoding": face_encoding,
            "face_location_fr": face_locations[0],
            "face_detected_cv": face_detected_cv
        }

    except Exception as e:
        return False, str(e)

def compare_face_encodings(known_encoding, unknown_encoding, tolerance=0.6):
    if known_encoding is None or unknown_encoding is None:
        return False, 1.0
    distance = face_recognition.face_distance([known_encoding], unknown_encoding)[0]
    return distance <= tolerance, distance

def download_and_analyze_image(image_url):
    try:
        response = requests.get(image_url, timeout=5)
        response.raise_for_status()
        img_bytes = response.content
        img_io = io.BytesIO(img_bytes)
        img_pil = PIL.Image.open(img_io).convert('RGB')
        img_np = np.array(img_pil)

        face_locations = face_recognition.face_locations(img_np)
        if not face_locations:
            return None, None

        face_encoding = face_recognition.face_encodings(img_np, face_locations)[0]

        return face_locations[0], face_encoding

    except requests.exceptions.RequestException:
        return None, None
    except Exception:
        return None, None

def find_image_urls_in_soup(soup, base_url):
    image_urls = []
    for img_tag in soup.find_all('img'):
        src = img_tag.get('src')
        if src:
            full_url = urllib.parse.urljoin(base_url, src)
            image_urls.append(full_url)
    return image_urls

def perform_search(platform, name):
    results = []
    image_urls_found = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/500.36'
    }
    query = f'"{name}" personal information contact social media profiles biography date of birth location email phone'
    url = ''
    if platform == 'google':
        url = f'https://www.google.com/search?q={urllib.parse.quote(query)}'
    elif platform == 'duckduckgo':
        url = f'https://duckduckgo.com/html/?q={urllib.parse.quote(query)}'
    else:
        return [], []

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = bs4.BeautifulSoup(response.text, 'html.parser')

        if platform == 'google':
             base_url = 'https://www.google.com/'
             for g in soup.find_all('div', class_='tF2CMy'):
                link_tag = g.find('a')
                if link_tag:
                    link = link_tag['href']
                    title_tag = g.find('h3')
                    title = title_tag.text if title_tag else 'N/A'
                    snippet_tag = g.find('div', class_='lEBKkf') or g.find('div', class_='VwiC3b')
                    snippet = snippet_tag.text if snippet_tag else 'No snippet available.'

                    results.append({
                        'source': 'Google',
                        'title': title,
                        'link': link,
                        'snippet': snippet
                    })
                    if link and link.startswith('http'):
                         try:
                             page_response = requests.get(link, headers=headers, timeout=5)
                             if page_response.status_code == 200:
                                 page_soup = bs4.BeautifulSoup(page_response.text, 'html.parser')
                                 image_urls_found.extend(find_image_urls_in_soup(page_soup, link))
                                 results[-1]['full_text_snippet'] = page_soup.get_text(separator=' ', strip=True)[:2000]
                         except requests.exceptions.RequestException:
                             pass


        elif platform == 'duckduckgo':
             base_url = 'https://duckduckgo.com/'
             for r in soup.select('#links .result'):
                title_tag = r.select_one('.result__a')
                if title_tag:
                    title = title_tag.text
                    link = title_tag['href']
                    snippet_tag = r.select_one('.result__snippet')
                    snippet = snippet_tag.text if snippet_tag else 'No snippet available.'

                    results.append({
                        'source': 'DuckDuckGo',
                        'title': title,
                        'link': link,
                        'snippet': snippet
                    })
                    if link and link.startswith('http'):
                         try:
                             page_response = requests.get(link, headers=headers, timeout=5)
                             if page_response.status_code == 200:
                                 page_soup = bs4.BeautifulSoup(page_response.text, 'html.parser')
                                 image_urls_found.extend(find_image_urls_in_image_urls_found(page_soup, link))
                                 results[-1]['full_text_snippet'] = page_soup.get_text(separator=' ', strip=True)[:2000]
                         except requests.exceptions.RequestException:
                             pass

    except requests.exceptions.RequestException:
        pass

    return results, image_urls_found

def extract_info_from_text(text):
    extracted = {}

    patterns = {
        'emails': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phones': r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        'social_media_facebook': r'(?:https?://)?(?:www\.)?facebook\.com/[A-Za-z0-9_.]+',
        'social_media_twitter': r'(?:https?://)?(?:www\.)?twitter\.com/(?:#!/)?[A-Za-z0-9_]+',
        'social_media_linkedin': r'(?:https?://)?(?:www\.)?linkedin\.com/in/[A-Za-z0-9_-]+',
        'social_media_instagram': r'(?:https?://)?(?:www\.)?instagram\.com/[A-Za-z0-9_.]+',
        'social_media_github': r'(?:https?://)?(?:www\.)?github\.com/[A-Za-z0-9_-]+',
        'social_media_pinterest': r'(?:https?://)?(?:www\.)?pinterest\.com/[A-Za-z0-9_-]+',
        'social_media_tumblr': r'(?:https?://)?(?:www\.)?tumblr\.com/blog/[A-Za-z0-9_-]+',
        'social_media_youtube': r'(?:https?://)?(?:www\.)?youtube\.com/(?:user/|channel/)?([a-zA-Z0-9_-]+)',

        'possible_birth_dates': [
            r'\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b',
            r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?),?\s+\d{4}\b',
            r'\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b',
            r'\b(?:born on)\s+([\w\s,.\-/]+)\b'
        ],
        'possible_locations': [
            r'\b(?:born in|from|lives in|based in|resides in)\s+([A-Z][a-z]+(?:,\s*[A-Z]{2})?(?:,\s*[A-Z][a-z]+)?(?:,\s*[A-Za-z\s]+)?)\b',
            r'\b(?:located in)\s+([A-Z][a-z]+(?:,\s*[A-Z]{2})?(?:,\s*[A-Z][a-z]+)?(?:,\s*[A-Za-z\s]+)?)\b',
            r'\b(?:(?:attended|studied) .*? in)\s+([A-Z][a-z]+(?:,\s*[A-Z]{2})?(?:,\s*[A-Z][a-z]+)?(?:,\s*[A-Za-z\s]+)?)\b',
        ],
        'possible_gender_cues': r'\b(he|him|his|she|her|hers|male|female)\b',
        'possible_employers': r'\bat\s+([A-Z][a-zA-Z\s&,.-]+?)\s+(?:Inc|Ltd|LLC|Co|Corp|\b(?:company|group))\b',
        'possible_education': r'\b(?:studied|attended)\s+([A-Z][a-zA-Z\s&,.-]+?)\s+(?:University|College|School)\b',
        'possible_relationships': r'\b(?:married to|spouse of|partner of)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\b',
        'possible_usernames': r'(?<![@\w\-.])(?:\b[A-Za-z0-9_-]{4,20}\b)(?!@)',
    }

    for key, pattern in patterns.items():
        if isinstance(pattern, list):
            extracted[key] = []
            for p in pattern:
                extracted[key].extend(re.findall(p, text, re.IGNORECASE))
            if key in ['possible_locations', 'possible_birth_dates']:
                 extracted[key] = [item[0] if isinstance(item, tuple) else item for item in extracted[key]]
            extracted[key] = list(set(extracted[key]))
        else:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                if key == 'possible_gender_cues':
                     found_genders = set()
                     for match in matches:
                         lower_match = match.lower()
                         if lower_match in ['he', 'him', 'his', 'male']:
                             found_genders.add('Male')
                         elif lower_match in ['she', 'her', 'hers', 'female']:
                             found_genders.add('Female')
                     extracted[key] = list(found_genders)
                elif key == 'social_media_youtube':
                     extracted[key] = [m[0] for m in matches]
                else:
                    extracted[key] = list(set(matches))

    return extracted

if __name__ == '__main__':
    person_name = input("Enter full name (lowercase, support space): ").lower()
    image_filename = input("Enter the filename of the face image (must be in this folder): ")
    image_path = os.path.join(os.getcwd(), image_filename)

    face_analysis_success, face_analysis_result = analyze_face(image_path)

    if not face_analysis_success:
        print(f"Face analysis failed: {face_analysis_result}")
        face_data = None
        input_face_encoding = None
    else:
        print("Face analysis successful.")
        face_data = face_analysis_result
        input_face_encoding = face_data.get("face_encoding")

    all_results = []
    all_image_urls = []

    print("\nInitiating concurrent search using 6 processes...")
    start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        futures = {}
        for i in range(3):
            futures[executor.submit(perform_search, 'google', person_name)] = 'google'
        for i in range(3):
            futures[executor.submit(perform_search, 'duckduckgo', person_name)] = 'duckduckgo'

        for future in concurrent.futures.as_completed(futures):
            platform = futures[future]
            try:
                results, image_urls = future.result()
                all_results.extend(results)
                all_image_urls.extend(image_urls)
                print(f" {platform.capitalize()} search finished.")
            except Exception as exc:
                print(f" {platform.capitalize()} search generated an exception: {exc}")

    search_end_time = time.time()
    print(f"Search completed in {search_end_time - start_time:.2f} seconds.")

    print("\nProcessing Search Results (Text and Images):")
    extracted_info = {}
    for res in all_results:
        if 'error' in res:
            continue
        text_to_analyze = res.get('snippet', '') + ' ' + res.get('title', '') + ' ' + res.get('link', '') + ' ' + res.get('full_text_snippet', '')
        info = extract_info_from_text(text_to_analyze)
        for key, values in info.items():
            if key not in extracted_info:
                extracted_info[key] = set()
            for value in values:
                extracted_info[key].add(value)

    potential_image_matches = []
    if input_face_encoding is not None:
        unique_image_urls = list(set(all_image_urls))
        print(f" Attempting to download and analyze {len(unique_image_urls)} unique images found online using 2 threads for face matching.")

        image_analysis_start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as image_executor:
            image_analysis_futures = {image_executor.submit(download_and_analyze_image, url): url for url in unique_image_urls}
            for future in concurrent.futures.as_completed(image_analysis_futures):
                url = image_analysis_futures[future]
                try:
                    face_location, face_encoding = future.result()
                    if face_encoding is not None:
                        is_match, distance = compare_face_encodings(input_face_encoding, face_encoding, tolerance=0.6)
                        if is_match:
                             potential_image_matches.append({'url': url, 'distance': distance})
                             print(f"  Potential face match found (Distance: {distance:.4f}): {url}")
                except Exception:
                    pass

        image_analysis_end_time = time.time()
        print(f"Image analysis completed in {image_analysis_end_time - image_analysis_start_time:.2f} seconds.")

    print("\n--- OSINT Report ---")

    print("\nPotential Extracted Information (from text analysis of search results and snippets):")
    if extracted_info:
        for key, values in extracted_info.items():
            print(f"{key.replace('_', ' ').title()}:")
            if values:
                for value in values:
                    print(f" - {value}")
            else:
                print(" - None found")
    else:
        print("No specific information patterns found in text results.")

    print("\nPotential Image Correlations (based on face recognition similarity, distance <= 0.6):")
    if potential_image_matches:
        sorted_matches = sorted(potential_image_matches, key=lambda x: x['distance'])
        for match in sorted_matches:
             print(f" - Distance {match['distance']:.4f}: {match['url']}")
    else:
        print("No strong potential face matches found in online images.")

    print("\nNote: This is an automated process based on publicly available information and face similarity algorithms. Verification of identity and personal details requires further investigation.")
