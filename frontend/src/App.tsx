import { Demo } from "./components/Demo";
import { Footer } from "./components/Footer";
import { Hero } from "./components/Hero";
import { Metrics } from "./components/Metrics";
import { Nav } from "./components/Nav";
import { Pipeline } from "./components/Pipeline";
import { Problem } from "./components/Problem";

/** The FaceProof landing page: story, proof, and the live demo on one scroll. */
export function App() {
  return (
    <>
      <Nav />
      <main>
        <Hero />
        <Problem />
        <Pipeline />
        <Metrics />
        <Demo />
      </main>
      <Footer />
    </>
  );
}
