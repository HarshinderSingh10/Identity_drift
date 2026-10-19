/**
 * References exactly as printed in the paper (IEEE_major_project.pdf, "REFERENCES").
 * Nothing has been added, completed or corrected. Items needing human checks are
 * listed in docs/CONTENT_VERIFICATION.md.
 */

export interface Reference {
  /** Number as printed in the paper, e.g. 3 for "[3]". */
  n: number
  text: string
}

export const references: Reference[] = [
  {
    n: 1,
    text: 'F. Schroff, D. Kalenichenko, and J. Philbin, “Facenet: A unified embedding for face recognition and clustering,” in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2015, pp. 815–823.',
  },
  {
    n: 2,
    text: 'H. Wang, Y. Wang, Z. Zhou, X. Ji, D. Gong, J. Zhou, Z. Li, and W. Liu, “Cosface: Large margin cosine loss for deep face recognition,” in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp. 5265–5274.',
  },
  {
    n: 3,
    text: 'J. Deng, J. Guo, N. Xue, and S. Zafeiriou, “Arcface: Additive angular margin loss for deep face recognition,” in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2019, pp. 4690–4699.',
  },
  {
    n: 4,
    text: 'K. Ricanek and T. Tesafaye, “Morph: A longitudinal image database of normal adult age-progression,” in Proc. 7th International Conference on Automatic Face and Gesture Recognition. IEEE, 2006, pp. 341–345.',
  },
  {
    n: 5,
    text: 'S. Moschoglou, A. Papaioannou, C. Sagonas, J. Deng, D. Kollias, and S. Zafeiriou, “Agedb: The first manually collected, in-the-wild age database,” in Proc. IEEE Conference on Computer Vision and Pattern Recognition Workshops, 2017, pp. 51–60.',
  },
  {
    n: 6,
    text: 'U. Park, Y. Tong, and A. K. Jain, “Age-invariant face recognition,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 32, no. 5, pp. 947–954, 2010.',
  },
  {
    n: 7,
    text: 'H. Han, C. Otto, and A. K. Jain, “Age estimation from face images: Human vs. machine performance,” in Proc. International Conference on Biometrics (ICB), 2013, pp. 1–8.',
  },
]
