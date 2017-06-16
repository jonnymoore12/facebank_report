# facebank

A face recognition application conceived and developed for the Financial Services event Banking Redefined, facebank was created as a prototype application to aid with registration at the event. It was additionally tasked with the responsibility of opening up conversations on how publicly accessible personal data and biometrics can be easily used in tandem to create exciting technologies in financial services and beyond.

The application was developed by a small team of beached ThoughtWorkers (originally Jonny Moore, Linda Roy and Yuki Takeuchi, based on some initial code from Isa Goksu) in approximately one month using Python and the open source library OpenCV.

The brainchild of Isa, facebank was pitched as an application that should use facial recognition to identify Banking Redefined attendees and potentially go on to recommend upcoming ThoughtWorks events tailored to their individual interests and experience (again, all publicly available information).

Here come some highlights, code snippets and discoveries from the project...

## SETUP / FIRST IMPRESSIONS
face_recognition, supplied by OpenCV (Open Source Computer Vision), relies on a small cluster of dependencies, including Dlib, CMake and Boost (some of which are written in C++). For those interested in playing around with this, feel free to email jmoore@thoughtworks.com for further information. Our repo has supplied a bootstrap which should streamline the setup process (thanks to Dianing Yudono for her help here).
```
import face_recognition
import cv2
import numpy as np
```

In simple terms, facebank worked as thus:
1. Upload 'known people' (at least one image of each of their face per person with a corresponding name) to the app
2. Create a face encoding for each image
3. Get a reference to the webcam and scan for faces
4. Compare each face captured by the webcam with known faces and determine a best match for each captured face
5. Output some feedback, minimally the name corresponding to the best match for each face.

For detailed information on what's happening behind the scenes with OpenCV's Face Recognition, check out:
[OpenCV Face Recognition](http://docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html)


First we supplied facebank with known people, got a reference to the webcam and initialized some variables:
```
video_capture = cv2.VideoCapture(0)

face_locations = []
face_encodings = []
face_names = []
```
We run the application in a while loop, continually capturing a series of images from the webcam, with the interval between frames simply being determined by the time it takes for one loop of the code to execute:
```
while True:
    ret, frame = video_capture.read()
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    face_names = []
        for idx, face_encoding in enumerate(face_encodings):

            # measure euclidean distances of each image with the ones we know of
            distances = np.linalg.norm(known_faces - face_encoding, axis=1)
            name = names[np.argmin(distances, axis=0)]
            face_names.append(name)
```
So we have an array of face_names corresponding to the 'best match' i.e. the smallest Euclidean distance between captured face and known face.

For feedback for the user, we opted for facebank to draw a rectangle around each face it identifies and to crudely display the name. Here comes more code:
```
for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)
```
![imageedit_8_6008069884](https://user-images.githubusercontent.com/18581870/27182672-f3fa0152-51d3-11e7-8abd-de703ba357e8.png)

It's worth emphasizing that facebank simply looks for the single 'best match' or shortest mathematical distance between captured face and known face. In other words, if the only face facebank is prepped with is George Clooney's, anyone captured by the app will automatically be identified as George Clooney. Despite how flattering this approach may (arguably) have been, we of course needed to test that the app would hold up and be accurate when having to distinguish between the approximately 200 guests that would be invited to Banking Redefined...

<img width="942" alt="emem2" src="https://user-images.githubusercontent.com/18581870/27183740-9ea1fb20-51d7-11e7-87f2-b7de617d6893.png">

## ACCURACY AND SCALE EXPERIMENTS
For testing purposes, we filled facebank with London ThoughtWorkers and ran accuracy tests using beach and Ops folks as our test subjects. This preliminary testing suggested a raw accuracy in the region of 85% when the app was loaded with 100 known faces (the remaining 15% were either completely misidentified or more commonly prompted facebank into indecisiveness with the displayed name flickering between the correct name and one or more alternative names as the app looped through execution).

<img width="1273" alt="fb2" src="https://user-images.githubusercontent.com/18581870/27093429-89162986-505e-11e7-9acf-a3f0cdf5d7ee.png">
<img width="1272" alt="fb4" src="https://user-images.githubusercontent.com/18581870/27187401-092ecbca-51e3-11e7-8901-243a141d36d3.png">

Some recurring problem areas:
 - Low quality images
 - Low resemblance between subject as they look now and on the day compared with the uploaded images of them
 - Glasses (the test subject wearing glasses in person but not in the image uploaded to facebank or vice versa)
 - Poor lighting of the subject as captured on webcam (e.g. heavily backlit)
 - Faces with dark skin tones are susceptible to not being recognized as faces at all in low light (though it should be noted that, at least for our small sample size, it was generally not especially problematic correctly identifying people of dark skin tones when they were well lit).

#### Accuracy Improvements
Whilst the out of the box accuracy was deemed reasonable for our purposes and scale (given the prototype nature of the app), we began a period of research into methods to boost the accuracy further, retesting each in isolation to assess any potential accuracy improvements. Some of the things we trialled:
 - Multiple and potentially many images of each attendee (though since this is accompanied by a performance hit, we concluded that one or two good images were preferable instead)
 - Converting images to grayscale
 - Histogram equalization to smooth out lighting differences
 - "Good" images of attendees (see below).

The only one of these techniques which yielded any statistically significant improvement to accuracy was ensuring that the original images we fed to facebank were "good" images, defined as such:
 - Images should be of sufficiently high resolution (no pixelation of features)
 - None of the face should be obscured or omitted
 - Face should be approximately head on (OpenCV rather intelligently will account for faces captured at a slight angle, but it is only so lenient; faces at extreme angles will not register as faces at all)
 - Images should not be heavily underexposed, blown out, suffer from excessive noise, etc.

Whilst some of these methods may have led to perceptible accuracy improvements if tested at a much larger scale (i.e. testing over thousands of known faces), the conclusion was that, for our purposes, we should concentrate on uploading "good" images of each attendee. We would collect LinkedIn profile pictures of attendees, with a Google search or a peek at company websites as a backup. We should note that for some attendees, there were no suitable publicly available images and facebank would simply not be able to identify them; our accuracy at the event would thus suffer an automatic hit.

#### I Always Remember A Face
All of the methods to potentially improve the accuracy outlined above were based on facebank re-evaluating the webcam capture and determining a best match for <em>every single captured iteration of execution on a continuous basis</em>. We deemed the app to be accurate in the event that it successfully identified a face for the entire time they were captured (or at least the vast majority of the time with little indecisiveness), whereas in our failing cases, the labelled name would flicker between multiple options.

<em>Could we program our application to make one decision and stick to it? Better yet, could we improve the accuracy of that single decision?</em>

One issue with the setup was that between each loop of execution there was nothing connecting faces in consecutive iterations i.e. if two faces are captured in consecutive iterations, say Hernan's and Dianing's, there is nothing connecting Hernan in iteration 1 to Hernan in iteration 2 or Dianing in iteration 1 with Dianing in iteration 2, there are simply two "new" faces with each pass through the loop.
<img width="1274" alt="fb1" src="https://user-images.githubusercontent.com/18581870/27125879-a9502b04-50ed-11e7-96be-88abb86567b1.png">
By considering the overlapping intersecting face areas in consecutive iterations we were able to establish a continuity between each iteration (Hernan is Hernan, Dianing is Dianing).

<img width="558" alt="intersecting_areas" src="https://user-images.githubusercontent.com/18581870/27172133-0ae9b8d0-51ac-11e7-8969-0715fdda3987.png">

We achieved this by calculating, for each rectangle in the current frame, the rectangle in the previous frame which had the most overlap with it:
```
def area(rectangle1, rectangle2):
    (left1, top1, right1, bottom1) = rectangle1
    (left2, top2, right2, bottom2) = rectangle2

    intersection_area = 0
    dx = min(right1, right2) - max(left1, left2)
    dy = min(top1, top2) - max(bottom1, bottom2)

    if (dx >= 0) and (dy >= 0):
        intersection_area = dx * dy

    return intersection_area

def best_match_index(new_location, prev_locations):
    greatest_intersection_area = 0
    index = -1
    for idx, location in enumerate(prev_locations):
        intersection_area = area(new_location, location)
        if (intersection_area > greatest_intersection_area):
            greatest_intersection_area = intersection_area
            index = idx
    return index
```

With this in place, we experimented with various statistical methods of identifying a face and essentially taking an average, but over <em>a number of iterations of execution</em>. facebank could make a decision over time and stick to it for the entire time a face remains on screen.

#### Who Invited You?
And what of 'George Clooney'? For uninvited guests, should facebank simply default to the 'best known match' or could we program in the ability to confidently distinguish between 'known' faces and 'surprise' faces?

By analysing the Euclidean distances between a captured face and each known face (essentially the mathematically perceived resemblances between the two), specifically looking at the distances to the best few matches, we noticed that the distance from Keerthana's face as captured by the webcam to the stored headshot of Keerthana is not only the smallest, but, furthermore stands out from the crowd, so to speak (as you might expect).

<img width="637" alt="nearest_neighbours" src="https://user-images.githubusercontent.com/18581870/27178047-9a28d47c-51c0-11e7-8678-7d1472da5987.png">

If we omit the distance derived from a comparison with Keerthana's uploaded headshot (as if facebank has no knowledge of her), the remaining minimum distances are considerably more bunched up; significantly these results were not atypical.

<img width="694" alt="nearest_neighbours2" src="https://user-images.githubusercontent.com/18581870/27178300-9fef18ca-51c1-11e7-84a2-e6f4a61a1453.png">

In conclusion, if d2-d1 is above a particular threshold, the captured face bears a close enough resemblance to one of the known faces and we can determine that said face belongs to an invited attendee. Conversely, if d2-d1 is too small, we establish, with a fair degree of certainty, that we have an uninvited on our hands.

Whilst this feature was exciting and largely successful, it came with two big caveats. No matter how we determined the d2-d1 threshold (normalized the sample data and factored in the standard deviation, etc.), the tradeoff was that in order to capture the vast majority of uninvited guests, some invited guests would inevitably be incorrectly labelled as uninvited (there would be overlap between the two). Furthermore, with the extra computation going on, facebank additionally suffered significant performance issues, with the rendered output refreshing at a decidedly lower frame rate.

Even our experiments into remembering faces, making decisions and sticking to identifications, hit the performance of the app with no perceived benefit to accuracy. Sometimes captured faces bear greater resemblance (crucially, consistently so) to uploaded images of others than to their own uploaded images, no matter how you rearrange the numbers. Consequently, most of this experimental code, as fun as it was to cook up, was not incorporated into our final app.

I would, however, not definitively rule out using some of these techniques, perhaps in conjunction with additional methods, as a means of potentially driving up accuracy or indeed of arriving at identification and presenting it to users in a revised format.

## GUI and Mapping To Events
Since we wanted to be able to not only identify attendees but also map them to upcoming ThoughtWorks events, we looked into building a more user-friendly GUI to present our recommendations.

We started matching attendees to relevant future events, storing their data in a data.json:
```
...

"Linda Roy": {
    "company": "ThoughtWorks",
    "events": [
        "Banking Redefined"
    ],
    "photos": [
        "lindaRoy2",
        "lindaRoy"
    ],
    "position": "QA"
},

...
```
and experimented with packaging the app up in Flask. However, deciding that on the day of the event we would simply need to run facebank locally on our laptops, one good GUI solution we found was to simply create it using OpenCV itself:

<img width="1275" alt="edyta" src="https://user-images.githubusercontent.com/18581870/27229244-c892a7a6-52a2-11e7-9c03-001f9a6673ad.png">

Despite building this feature, with a limited number of confirmed future ThoughtWorks events, we omitted this from the final version of the app, instead opting for this simple display as seen via a selfie taken at the event itself:

![image uploaded from ios 1](https://user-images.githubusercontent.com/18581870/27184208-5b9d9512-51d9-11e7-91ea-ee4003f6b41a.jpg)

At this point, we also checked the accuracy and performance of the app, uploading 200 known people to ensure the app would function as intended for Banking Redefined itself.

We hardware tested facebank, connecting it to HD external cameras and monitors. We also trialled the application at the event venue, Banking Hall in London, determining that due to the dim lighting throughout the interiors (I mean, we all love a touch of mood lighting, but it is dark in there), we would need additional lighting to ensure attendees were sufficiently lit.

## PRODUCTION READY & BANKING REDEFINED
With the event drawing near, we:
 - Updated facebank, uploading a complete and final list of all registered attendees (ThoughtWorks and external)
 - Tested edge cases (ran the app continuously for hours, overloaded it with as many people as the external cameras could possibly capture, etc.)
 - Created Python scripts to ensure the spelling of all names and information stored in the app was correct and consistent.

Thanks to Hernan Carrizo and Caitlin Gulliford for their help here.

### Banking Redefined
![banking_hall](https://user-images.githubusercontent.com/18581870/27137709-7e274524-5116-11e7-9da3-475e0c4bf7ba.jpeg)
facebank was used as a non-intrusive addition to registration at the event. As attendees approached a registration desk to collect pre-printed name badges (with the monitors aimed at us, rather than the attendees themselves), we were able to welcome them by their names before they announced them themselves in some cases (eliciting surprised reactions). This, however, was often not possible as many attendees would state their own names as they arrived, denying us the opportunity.

During the afternoon at the event, we were able to setup facebank and a monitor for attendees to approach and engage with at their own pace. The app correctly identified, as a best estimate, ~85% of attendees who were all some combination of intrigued, surprised and excited by it.

### FUTURE facebank
facebank was merely a small tech prototype addition to Banking Redefined, but there is so much scope to build on this and potentially roll out something much more substantial to be used at other events or to be incorporated into heavier, non event-related applications.

Here come some event-related ideas:
1. Touch-screen self-service registration: the app could offer 3 candidate best matches for each attendee, with an option for attendees to select "other" and manually select their name, ensuring 100% accuracy.
2. Face recognition could be combined with an additional type of biometric (voice, gait, fingerprint, etc.) for much greater accuracy
3. At Banking Redefined we trialled the concept of PDRs (Personal Data Receipts) that were manually emailed to each attendee. An app that is 100% accurate could automate this process, with emails being fired off by successful identification.

ThoughtWorks are working elsewhere in the biometrics space (see the [smart mirror](https://www.thoughtworks.com/clients/standardcharteredbank) in Singapore as one example). Hopefully other projects can pick up the baton and delve further into these exciting technologies.
