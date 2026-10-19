import { ButtonLink } from '../components/ui/ButtonLink.tsx'
import { PageHeader } from '../components/ui/PageHeader.tsx'
import { Section } from '../components/ui/Section.tsx'
import { usePageTitle } from '../hooks/usePageTitle.ts'

export default function NotFound() {
  usePageTitle('Page not found')

  return (
    <>
      <PageHeader
        eyebrow="404"
        title="Page not found"
        description="The page you requested does not exist on this site."
      />
      <Section>
        <ButtonLink to="/">Back to home</ButtonLink>
      </Section>
    </>
  )
}
