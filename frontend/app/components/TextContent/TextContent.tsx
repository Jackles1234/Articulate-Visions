import React from "react";
import style from './style.module.css'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faGithub, faLinkedin } from '@fortawesome/free-brands-svg-icons';

const TextContent: React.FC = () => {
    return (
        <>
            <div id="main-part">
                <div className="first_row">
                    <h2> A Technical Deep Dive into Text-to-Image AI
                    </h2>
                    <p>
                        The ability to weave captivating visuals from the tapestry of language
                        is no longer science fiction. Text-to-image AI is revolutionizing creative
                        expression, empowering users to bridge the gap between imagination and
                        digital canvas. But what lies beneath the hood of this transformative
                        technology? Let&apos;s embark on a technical exploration, dissecting the core
                        functionalities and the fascinating processes that bring your words to life
                        as visuals.
                    </p>
                </div>

                <div className="second_row">
                    <h2> Deep Learning Models
                    </h2>
                    <p>
                        At the heart of text-to-image AI lie deep learning models, specifically
                        convolutional neural networks (CNNs) trained on massive datasets of text-image
                        pairs. These datasets act as the Rosetta Stone for the AI, enabling it to
                        decipher the complex relationships between textual descriptions and their
                        corresponding visual representations. When you provide a text prompt, the
                        model dissects your words, identifying key elements, objects, and their spatial
                        relationships. This extracted information serves as the foundation for the
                        image generation process.
                    </p>
                    <ul>
                        <li>
                            Encoder Networks: These CNNs process your text description, extracting a
                            high-dimensional vector representation that captures the semantic meaning
                            and relationships within your words.
                        </li>
                    </ul>
                </div>

                <div className="third_row">
                    <h2> From Noise to Form: Demystifying Diffusion Models
                    </h2>
                    <p>
                        One of the most prominent approaches in text-to-image AI is diffusion.
                        Diffusion models operate by gradually refining an initial state of noise
                        into a coherent image that aligns with your text description. Here&apos;s a
                        breakdown of the diffusion process:
                    </p>
                    <ul>
                        <li>
                            Guiding with Text: Your text description is embedded into a latent space
                            using another CNN (text encoder). This embedding is then incorporated into
                            the diffusion process, guiding the model towards generating an image that
                            reflects the semantic content of your words.
                        </li>
                        <li>
                            Iterative Refinement: The core of the diffusion process lies in a series of
                            denoising steps. At each step, the model predicts the &quot;cleaner&quot; version of
                            the current noisy image, effectively removing noise and introducing image
                            details based on the embedded text information.
                        </li>
                        <li>
                            Predicting the Next Step: The model utilizes a U-Net like architecture to
                            predict the &quot;denoised&quot; version of the current image. This prediction step
                            leverages the embedded text information to ensure the denoised image aligns
                            with your description.
                        </li>
                        <li>
                            The Unveiling: After a predetermined number of denoising steps, the model
                            outputs the final image - a visual representation meticulously crafted from
                            your text prompt and guided by the iterative diffusion process.
                        </li>
                    </ul>
                </div>

                <div className="fourth_row">
                    <h2> Exploring Alternative Text-to-Image Approaches
                    </h2>
                    <p>
                        While diffusion models are currently at the forefront,
                        other text-to-image approaches offer unique functionalities:
                    </p>
                    <ul>
                        <li>
                            Autoencoders: These models employ a two-part architecture: an encoder and
                            a decoder. The encoder compresses your text description into a latent space
                            representation, capturing its essence. The decoder then utilizes this latent
                            representation to generate an image reflecting the encoded information. This
                            approach excels at capturing the core semantic meaning of your text prompt and
                            translating it into a visually coherent image.
                        </li>
                        <li>
                            Attention-Based Models: These models incorporate attention mechanisms
                            that focus on specific parts of the generated image based on the relative
                            importance of different elements in your text prompt. Imagine the model
                            strategically allocating its resources to render crucial details you mentioned
                            in your description, like a flowing mane on a lion or the intricate scales of
                            a dragon. Attention-based models are particularly adept at generating images
                            with a high degree of fidelity to the specific details mentioned in the text prompt.
                        </li>
                        <li>
                            Generative Adversarial Networks (GANs): These models involve two competing
                            neural networks: a generator and a discriminator. The generator strives
                            to produce an image that aligns with your text description and artistic
                            style preferences. The discriminator, acting as a discerning critic,
                            meticulously evaluates the generated image, providing feedback to the
                            generator. Through this adversarial training process, the GANs refine the
                            image iteratively, pushing it closer to a visually compelling and semantically
                            accurate representation of your text prompt.
                        </li>
                    </ul>
                </div>
                <div className="fifth_row">
                    <h2> Explore the Cutting Edge:
                    </h2>
                    <p>
                        State-of-the-Art Models: Stay updated on the latest advancements by exploring these cutting-edge
                        text-to-image models:
                    </p>
                    <ul>
                        <li>
                            Imagen by Google AI: [invalid URL removed]
                        </li>
                        <li>
                            DALL-E 2 by OpenAI: [invalid URL removed]
                        </li>
                    </ul>
                </div>
                <div className="sixth_row">
                    <h2> Who are we?:
                    </h2>
                    <p>
                        This was a project made as part of the AI major capstone at Drake university.
                    </p>
                    <ul className={`${style.aboutUL}`}>
                        <li>
                            <div style={{display: 'flex', alignItems: 'center', gap: '10px'}}>
                                <span style={{fontWeight: 'bold'}}>Jack Welsh</span>
                                <a href="https://github.com/Jackles1234" target="_blank" rel="noopener noreferrer">
                                    <FontAwesomeIcon icon={faGithub} size="2x" style={{ color: 'black' }} />
                                </a>
                                <a href="https://www.linkedin.com/in/jack-welsh-bb849b250/" target="_blank"
                                   rel="noopener noreferrer">
                                    <FontAwesomeIcon icon={faLinkedin} size="2x" style={{ color: 'black' }} />
                                </a>
                            </div>
                        </li>
                        <li>
                            <div style={{display: 'flex', alignItems: 'center', gap: '10px'}}>
                                <span style={{fontWeight: 'bold'}}>Cooper Brown</span>
                                <a href="https://github.com/cbrown987" target="_blank" rel="noopener noreferrer">
                                    <FontAwesomeIcon icon={faGithub} size="2x" style={{ color: 'black' }} />
                                </a>
                                <a href="https://www.linkedin.com/in/cbrown987/" target="_blank"
                                   rel="noopener noreferrer">
                                    <FontAwesomeIcon icon={faLinkedin} size="2x" style={{ color: 'black' }} />
                                </a>
                            </div>
                        </li>
                        <li>
                            <div style={{display: 'flex', alignItems: 'center', gap: '10px'}}>
                                <span style={{fontWeight: 'bold'}}>Conrad Ernst</span>
                                <a href="https://github.com/ConradErnst" target="_blank" rel="noopener noreferrer">
                                    <FontAwesomeIcon icon={faGithub} size="2x" style={{ color: 'black' }} />
                                </a>
                                <a href="https://www.linkedin.com/in/conradernst/" target="_blank"
                                   rel="noopener noreferrer">
                                    <FontAwesomeIcon icon={faLinkedin} size="2x" style={{ color: 'black' }} />
                                </a>
                            </div>
                        </li>
                    </ul>
                </div>
            </div>
        </>
    )
}

export default TextContent;