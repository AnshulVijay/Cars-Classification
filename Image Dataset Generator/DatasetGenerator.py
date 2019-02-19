from google_images_download import google_images_download
def scrape_google_images(imageName, count):
    response = google_images_download.googleimagesdownload()
    arguments= {"keywords":imageName, "limit":count,"print_urls":True,  "chromedriver":'C:\WebDrivers\chromedriver.exe', 'prefix':imageName}
    paths = response.download(arguments)
scrape_google_images('SUV',3330)
scrape_google_images('Hatchback',3330)
scrape_google_images('Sedan',3330)
