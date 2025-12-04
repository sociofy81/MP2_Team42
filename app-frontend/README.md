# AI Photoshop App

A mobile photo-relighting prototype built with **Expo + React Native**.
The app allows users to load an image, adjust lighting properties, apply relighting effects, and preview the results in real time.

## Features

- Import a photo from device storage
- Adjustable lighting intensity
- Adjustable hue with a gradient hue slider
- Thumbnail selector with add button
- Clean mobile UI based on Figma design
- Works in Expo Go with QR scanning
- Built-in safe area handling for iPhone notch/Dynamic Island

## Tech Stack

- **React Native**
- **Expo**
- **React Navigation**
- **expo-linear-gradient** (for gradient hue slider)
- **react-native-svg** + **react-native-svg-transformer** (for SVG icons)
- **@expo/vector-icons** (UI icons)
- **react-native-safe-area-context** (notch support)

## Project Structure

```sh
.
├── .gitignore
├── App.js
├── app.json
├── assets
│   ├── adaptive-icon.png
│   ├── favicon.png
│   ├── icon.png
│   ├── icons
│   │   ├── adjust.svg
│   │   ├── commits.svg
│   │   ├── down.svg
│   │   ├── home.svg
│   │   ├── layers.svg
│   │   ├── lightning.svg
│   │   ├── more.svg
│   │   ├── redo.svg
│   │   ├── size.svg
│   │   ├── type.svg
│   │   ├── undo.svg
│   │   └── up.svg
│   ├── images
│   │   ├── halloween.png
│   │   ├── logo.png
│   │   └── thumb.png
│   └── splash-icon.png
├── index.js
├── LICENSE
├── metro.config.js
├── package-lock.json
├── package.json
├── README.md
└── src
    ├── components
    │   ├── BottomNav.js
    │   ├── MainImage.js
    │   ├── SidePanel.js
    │   ├── SlidersPanel.js
    │   └── TopBar.js
    └── screens
        ├── EditorScreen.js
        └── HomeScreen.js
```

## Getting Started

### Install dependencies:

```sh
npm install
```

 ### Start the development server:
```sh
npx expo start
```

#### Run on a physical iOS device

-   Install **Expo Go** from the Apple App Store
    
-   Open Expo Go
    
-   Scan the QR code shown in your terminal or browser
    
-   The app will load instantly

#### Run on a physical Android device

-   Install **Expo Go** from the Google Play Store
    
-   Open Expo Go
    
-   Scan the QR code shown in your terminal or browser
    
-   The app will load instantly

### iOS Simulator (requires Xcode):
```sh
npx expo start --ios
```

### Android Simulator (requires Android Development Studio):
```sh
npm expo start --android
```

### Building a Production App:
#### Use Expo Application Services (EAS):

```sh
npm install -g eas-cli
eas build -p ios
eas build -p android
```

#### This generates `.ipa` and `.apk/.aab` files for distribution.

## Notes

-   SVG support requires `metro.config.js` with `react-native-svg-transformer`.    
-   All icons should be exported at **3x resolution** or as **SVG** to avoid graininess.
-   Layout uses absolute positioning for side panel to match the Figma design.

## Precaution

-   The IP address of the backend server needs to be changed so that the code could run well. For that, change the ip address in API_BASE
at the top section of `src/screens/EditorScreen.js` to the public IP address of the server running the backend (if it is your own computer,
a simple `ip addr` will also work well).
