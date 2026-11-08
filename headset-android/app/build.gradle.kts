import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
  id("com.android.application")
  id("org.jetbrains.kotlin.android")
}

android {
  namespace = "ai.megamind.quest"
  compileSdk = 36
  buildToolsVersion = "35.0.0"

  defaultConfig {
    applicationId = "ai.megamind.quest"
    minSdk = 32
    targetSdk = 34 // Horizon OS v209 is Android 14 (API 34)
    versionCode = 1
    versionName = "0.1"
  }

  buildTypes {
    release {
      isMinifyEnabled = false
      signingConfig = signingConfigs.getByName("debug")
    }
  }
  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
  }
  lint { abortOnError = false; checkReleaseBuilds = false }
}

kotlin { compilerOptions { jvmTarget = JvmTarget.JVM_17 } }

// No runtime dependencies beyond the Kotlin stdlib: plain Android Views, Camera2, AudioRecord,
// HttpURLConnection. Keeps the build small and fast. JUnit is test-only (host JVM).
dependencies {
  testImplementation("junit:junit:4.13.2")
}

// `./gradlew testDebugUnitTest -Pmegamind.url=http://127.0.0.1:8300` also POSTs the exact request
// bodies the app builds to a running Megamind server (otherwise those tests are skipped); add
// `-Pmegamind.token=<API_TOKEN>` (or MEGAMIND_API_TOKEN) for a server with auth on.
tasks.withType<Test>().configureEach {
  systemProperty("megamind.url", (project.findProperty("megamind.url") as String?) ?: System.getenv("MEGAMIND_URL") ?: "")
  systemProperty("megamind.token", (project.findProperty("megamind.token") as String?) ?: System.getenv("MEGAMIND_API_TOKEN") ?: "")
  outputs.upToDateWhen { false }
  testLogging { events("passed", "skipped", "failed"); showStandardStreams = true }
}
